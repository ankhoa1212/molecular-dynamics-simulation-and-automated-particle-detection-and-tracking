"""Manual crop tool — Optimized for LodeSTAR Training & YOLO Dataset Building.

Usage:
    python crop_tool.py [folder]

Workflow Modes:
    [LodeSTAR]  Focus on creating PNG snippets for model training. (Crops: ON by default)
    [YOLO]      Focus on full-frame bounding boxes for datasets. (Crops: OFF by default)

Core Controls:
    Left-click-drag   Draw a new crop rectangle
    Right-click-drag  Pan the image
    Scroll wheel      Zoom in / out (centered on cursor)
    Left / Right      Previous / next image
    R                 Reset zoom and pan to fit
    Ctrl+Z            Undo last action
    + / -             Zoom in / out (keyboard)
    ?                 Open Help & Shortcut dialog

Advanced Features:
    E                 Toggle EDIT MODE (move/resize existing boxes)
    Y                 Cycle YOLO VIEW (Box -> Point -> Both -> None)
    T                 Convert selected detection to LodeSTAR crop
    Delete/Backspace  Delete selected crop (in Edit Mode)

Export:
    [Accept All]      Convert all computer detections on frame to manual labels.
    [Export Dataset]  Generate a standard YOLOv8 folder with images and data.yaml.
"""

# pylint: disable=too-many-lines
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Optional

import shutil

import cv2
import numpy as np
from PIL import Image, ImageTk

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
CROPS_SUBDIR = "crops"
# Matches: crop_<stem>_y<YYYY>_x<XXXX>.png  (optional _N suffix for duplicates)
CROP_PATTERN = re.compile(r"^crop_(.+)_y(\d+)_x(\d+)(?:_\d+)?\.png$")
MIN_CROP_PX = 5
ZOOM_STEP = 1.15
ZOOM_MIN = 0.05
ZOOM_MAX = 30.0
HANDLE_RADIUS = 6  # display half-size of corner handle squares (canvas px)
HANDLE_HIT = 12  # hit-detection half-size for corner handles (canvas px)


class ToolTip:
    """A simple tooltip implementation for Tkinter."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            font=("tahoma", "9", "normal"),
        )
        label.pack(ipadx=1)

    def hide_tip(self, event=None):
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()


class CropTool:  # pylint: disable=too-many-instance-attributes
    """Main application window and logic for the crop tool."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Manual Crop Tool")
        self.root.geometry("1280x800")
        self.root.minsize(640, 480)

        # ── Data state ───────────────────────────────────────────────────────
        self.folder: Optional[Path] = None
        self.image_paths: list[Path] = []
        self.current_index: int = -1
        self.pil_image: Optional[Image.Image] = None
        self.cv_image: Optional[np.ndarray] = None
        self._image_cache: dict[Path, Image.Image] = {}
        self._cache_size: int = 10

        # ── Transform state ──────────────────────────────────────────────────
        # display_coord = image_coord * display_scale + offset + pan
        self.zoom_level: float = 1.0  # relative to fit-to-canvas scale
        self.pan_x: float = 0.0  # canvas-space additional offset
        self.pan_y: float = 0.0
        self.fit_scale: float = 1.0  # scale that fits image into canvas
        self.img_offset_x: float = 0.0  # letterbox margin
        self.img_offset_y: float = 0.0

        # ── Internal rendering ───────────────────────────────────────────────
        self._tk_img: Optional[ImageTk.PhotoImage] = None  # must hold reference
        self._img_id: Optional[int] = None
        self._resize_after_id: Optional[str] = None

        # ── Interaction state ────────────────────────────────────────────────
        self._drag_start: Optional[tuple[int, int]] = None
        self._rect_id: Optional[int] = None
        self._pan_start_canvas: Optional[tuple[int, int]] = None
        self._pan_start_offset: tuple[float, float] = (0.0, 0.0)

        # ── Annotation tracking ──────────────────────────────────────────────
        # Each entry: (x1, y1, x2, y2, Optional[Path])
        # Coordinates are image-space pixels. Path points to the PNG crop if it exists.
        self._manual_annots: list[tuple[int, int, int, int, Optional[Path]]] = []
        self._yolo_labels: list[tuple[int, int, int, int, int]] = []  # (x1, y1, x2, y2, class_id)
        self._yolo_view_mode: str = "both"  # 'box' | 'point' | 'both' | 'none'
        self.project_mode: str = "lodestar"  # 'lodestar' | 'yolo'
        self._save_crops_mode: bool = True  # Default to ON for LodeSTAR mode

        # -- Crosshair state (for WSL visibility) --
        self._mouse_pos: Optional[tuple[int, int]] = None
        self._cross_v_id: Optional[int] = None
        self._cross_h_id: Optional[int] = None
        self._undo_stack: list[dict] = []
        self._fixed_size: Optional[int] = None
        self._rect_cross_v_id: Optional[int] = None
        self._rect_cross_h_id: Optional[int] = None
        self._fixed_preview_id: Optional[int] = None

        # ── Edit mode state ──────────────────────────────────────────────────
        self._edit_mode: bool = False
        self._selected_indices: set[int] = set()  # Set of indices in self._manual_annots
        self._selected_type: str = "manual"  # 'manual' | 'yolo' (multi-select only for manual)
        self._edit_drag_mode: Optional[str] = None  # 'move' | 'resize_nw' | etc.
        self._edit_drag_start_canvas: Optional[tuple[int, int]] = None
        self._edit_drag_crops_orig: dict[int, tuple[int, int, int, int]] = {}  # idx -> orig_coords

        self._build_ui()
        self._bind_events()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Create the toolbar (two rows), canvas, and status bar."""
        self._status = tk.StringVar(value="Open a folder to begin.")
        self.progress_bar = ttk.Progressbar(self.root, mode="indeterminate")

        # ── Toolbar container (holds two rows) ───────────────────────────────
        toolbar_container = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        toolbar_container.pack(side=tk.TOP, fill=tk.X)

        # ── Row 1: Mode | File | Navigation | Zoom | Help ────────────────────
        row1 = tk.Frame(toolbar_container)
        row1.pack(side=tk.TOP, fill=tk.X)

        # Mode Selector
        mode_frame = tk.Frame(row1, bg="#333333", padx=2)
        mode_frame.pack(side=tk.LEFT, padx=5, pady=2)

        self._btn_mode_lodestar = tk.Button(
            mode_frame,
            text="LodeSTAR Mode",
            command=lambda: self.switch_project_mode("lodestar"),
            font=("Helvetica", 9, "bold"),
        )
        self._btn_mode_lodestar.pack(side=tk.LEFT, padx=1, pady=2)

        self._btn_mode_yolo = tk.Button(
            mode_frame,
            text="YOLO Mode",
            command=lambda: self.switch_project_mode("yolo"),
            font=("Helvetica", 9, "bold"),
        )
        self._btn_mode_yolo.pack(side=tk.LEFT, padx=1, pady=2)
        _sep(row1)

        # File
        self._btn_open = tk.Button(row1, text="Open Folder", command=self.open_folder)
        self._btn_open.pack(side=tk.LEFT, padx=4, pady=3)
        ToolTip(self._btn_open, "Select a folder containing extracted PNG/TIFF frames")
        _sep(row1)

        # Navigation
        self._btn_prev = tk.Button(
            row1, text="← Prev", command=self.prev_image, state=tk.DISABLED
        )
        self._btn_prev.pack(side=tk.LEFT, padx=2, pady=3)
        ToolTip(self._btn_prev, "Go to previous image (Left Arrow)")

        self._lbl_counter = tk.Label(row1, text="No images", width=14)
        self._lbl_counter.pack(side=tk.LEFT, padx=4)

        self._btn_next = tk.Button(
            row1, text="Next →", command=self.next_image, state=tk.DISABLED
        )
        self._btn_next.pack(side=tk.LEFT, padx=2, pady=3)
        ToolTip(self._btn_next, "Go to next image (Right Arrow)")
        _sep(row1)

        # Zoom
        self._btn_zoom_in = tk.Button(row1, text="Zoom In (+)", command=self.zoom_in)
        self._btn_zoom_in.pack(side=tk.LEFT, padx=2, pady=3)
        ToolTip(self._btn_zoom_in, "Increase magnification")

        self._btn_zoom_out = tk.Button(row1, text="Zoom Out (−)", command=self.zoom_out)
        self._btn_zoom_out.pack(side=tk.LEFT, padx=2, pady=3)
        ToolTip(self._btn_zoom_out, "Decrease magnification")

        self._btn_reset = tk.Button(row1, text="Reset Zoom (R)", command=self.reset_transform)
        self._btn_reset.pack(side=tk.LEFT, padx=2, pady=3)
        ToolTip(self._btn_reset, "Fit image to window (R)")

        # Help (anchored to the right of row 1)
        tk.Button(row1, text="Help (?)", command=self.show_help, bg="#2d5a27", fg="white").pack(
            side=tk.RIGHT, padx=4, pady=3
        )

        # ── Row 2: Edit actions | YOLO / crops toggles | Advanced ────────────
        row2 = tk.Frame(toolbar_container, bg=toolbar_container.cget("bg"))
        row2.pack(side=tk.TOP, fill=tk.X)

        # Undo
        tk.Button(row2, text="Undo (Ctrl+Z)", command=self.undo_last_crop).pack(
            side=tk.LEFT, padx=4, pady=3
        )
        _sep(row2)

        # Edit Mode + Delete
        self._btn_edit_mode = tk.Button(
            row2, text="Edit Mode (E)", command=self.toggle_edit_mode, relief=tk.RAISED
        )
        self._btn_edit_mode.pack(side=tk.LEFT, padx=2, pady=3)
        ToolTip(
            self._btn_edit_mode,
            "Switch between drawing new boxes and editing existing ones (E)",
        )

        self._btn_delete = tk.Button(
            row2,
            text="Delete Selected (Del)",
            command=self.delete_selected_crop,
            state=tk.DISABLED,
        )
        self._btn_delete.pack(side=tk.LEFT, padx=2, pady=3)
        ToolTip(self._btn_delete, "Remove the selected box (Delete)")
        _sep(row2)

        # YOLO view cycle (always visible — relevant in both modes)
        self._btn_yolo_mode = tk.Button(
            row2, text="YOLO: Box (Y)", command=self.toggle_yolo_view_mode
        )
        self._btn_yolo_mode.pack(side=tk.LEFT, padx=2, pady=3)
        ToolTip(self._btn_yolo_mode, "Cycle through different ways to see computer detections (Y)")

        # Crops toggle (LodeSTAR-specific; hidden in YOLO mode)
        self._btn_save_crops = tk.Button(
            row2, text="Crops: Off (C)", command=self.toggle_save_crops_mode
        )
        ToolTip(
            self._btn_save_crops,
            "Enable this to save PNG files automatically while drawing (C)",
        )
        _sep(row2)

        # Fixed Size
        self._btn_fixed_size = tk.Button(
            row2, text="Fixed Size: Off (F)", command=self.toggle_fixed_size
        )
        self._btn_fixed_size.pack(side=tk.LEFT, padx=2, pady=3)
        ToolTip(self._btn_fixed_size, "Force clicks/drags to a fixed square size (F)")
        _sep(row2)

        # Convert (always visible for use in Edit Mode)
        self._btn_convert = tk.Button(
            row2, text="Convert to Crop (T)", command=self.convert_selected, state=tk.DISABLED
        )
        self._btn_convert.pack(side=tk.LEFT, padx=2, pady=3)
        ToolTip(self._btn_convert, "Convert selected detection into a LodeSTAR crop PNG")

        # Accept All Detections (YOLO-specific; hidden in LodeSTAR mode)
        self._btn_accept_all = tk.Button(
            row2, text="Accept All Detections", command=self.convert_all_yolo
        )
        ToolTip(
            self._btn_accept_all,
            "Turn all computer detections on this frame into manual labels",
        )

        # Export Dataset (YOLO-specific; anchored right so it stays prominent)
        self._btn_export = tk.Button(
            row2,
            text="Export YOLO Dataset",
            command=self.export_yolo_dataset,
            bg="#005a9e",
            fg="white",
        )
        ToolTip(
            self._btn_export,
            "Finalize all labeled images into a ready-to-train YOLO dataset folder",
        )

        # Canvas
        self.canvas = tk.Canvas(self.root, bg="#1e1e1e", cursor="none", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Status bar
        tk.Label(
            self.root,
            textvariable=self._status,
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=("Helvetica", 9),
            padx=4,
        ).pack(side=tk.BOTTOM, fill=tk.X)

        # Initialize Mode (MUST be last after all widgets are created)
        self.switch_project_mode("lodestar")

    def _bind_events(self) -> None:
        root, canvas = self.root, self.canvas

        # Navigation
        root.bind("<Left>", lambda _: self.prev_image())
        root.bind("<Right>", lambda _: self.next_image())

        # Zoom / reset
        root.bind("<r>", lambda _: self.reset_transform())
        root.bind("<R>", lambda _: self.reset_transform())
        root.bind("<plus>", lambda _: self.zoom_in())
        root.bind("<equal>", lambda _: self.zoom_in())
        root.bind("<minus>", lambda _: self.zoom_out())

        # Undo
        root.bind("<Control-z>", lambda _: self.undo_last_crop())

        # Edit mode toggle / delete
        root.bind("<e>", lambda _: self.toggle_edit_mode())
        root.bind("<E>", lambda _: self.toggle_edit_mode())
        root.bind("<Delete>", lambda _: self.delete_selected_crop())
        root.bind("<BackSpace>", lambda _: self.delete_selected_crop())
        root.bind("<y>", lambda _: self.toggle_yolo_view_mode())
        root.bind("<Y>", lambda _: self.toggle_yolo_view_mode())
        root.bind("<c>", lambda _: self.toggle_save_crops_mode())
        root.bind("<C>", lambda _: self.toggle_save_crops_mode())
        root.bind("<t>", lambda _: self.convert_selected())
        root.bind("<T>", lambda _: self.convert_selected())
        root.bind("<f>", lambda _: self.toggle_fixed_size())
        root.bind("<F>", lambda _: self.toggle_fixed_size())

        # Scroll-to-zoom: Linux Button-4/5, Windows/Mac MouseWheel
        canvas.bind("<MouseWheel>", self._on_mousewheel)
        canvas.bind("<Button-4>", self._on_mousewheel)
        canvas.bind("<Button-5>", self._on_mousewheel)

        # Right-click pan
        canvas.bind("<ButtonPress-3>", self._on_pan_start)
        canvas.bind("<B3-Motion>", self._on_pan_drag)
        canvas.bind("<ButtonRelease-3>", self._on_pan_end)

        # Left-click rubber-band crop
        canvas.bind("<ButtonPress-1>", self._on_drag_start)
        canvas.bind("<B1-Motion>", self._on_drag_motion)
        canvas.bind("<ButtonRelease-1>", self._on_drag_end)

        # Window resize
        canvas.bind("<Configure>", self._on_canvas_configure)

        # Mouse motion for crosshair
        canvas.bind("<Motion>", self._on_mouse_move)
        canvas.bind("<Leave>", self._on_mouse_leave)

    # ── Edit mode ────────────────────────────────────────────────────────────

    def toggle_edit_mode(self) -> None:
        """Toggle between crop drawing and edit modes."""
        self._edit_mode = not self._edit_mode
        if not self._edit_mode:
            self._selected_indices = set()
            self._selected_type = "manual"
            self._edit_drag_mode = None
            self._edit_drag_start_canvas = None
            self._edit_drag_crops_orig = {}
            self._btn_delete.config(state=tk.DISABLED)
            self._btn_convert.config(state=tk.DISABLED)
        else:
            # Entering edit mode — ensure buttons start disabled
            self._btn_delete.config(state=tk.DISABLED)
            self._btn_convert.config(state=tk.DISABLED)
        self._btn_edit_mode.config(relief=tk.SUNKEN if self._edit_mode else tk.RAISED)
        self.canvas.config(cursor="arrow" if self._edit_mode else "crosshair")
        self._redraw()
        self._update_status()

    def switch_project_mode(self, mode: str) -> None:
        """Switch between LodeSTAR (crop-focused) and YOLO (box-focused) workflows."""
        self.project_mode = mode
        if mode == "lodestar":
            self._save_crops_mode = True
            self._btn_mode_lodestar.config(relief=tk.SUNKEN, bg="#00FF88", fg="black")
            self._btn_mode_yolo.config(relief=tk.RAISED, bg="#f0f0f0", fg="black")
        else:
            self._save_crops_mode = False
            self._btn_mode_lodestar.config(relief=tk.RAISED, bg="#f0f0f0", fg="black")
            self._btn_mode_yolo.config(relief=tk.SUNKEN, bg="#FF00FF", fg="white")

        self._update_ui_visibility()
        self._update_status()
        self._redraw()

    def _update_ui_visibility(self) -> None:
        """Hide/Show mode-specific buttons in row 2 based on current project mode.

        LodeSTAR mode: show "Crops: On/Off" toggle; hide YOLO-only actions.
        YOLO mode:     hide crop toggle; show "Accept All" and "Export Dataset".
        """
        if self.project_mode == "lodestar":
            # Insert the Crops toggle right after the YOLO-view button (its natural neighbour)
            self._btn_save_crops.pack(after=self._btn_yolo_mode, side=tk.LEFT, padx=2, pady=3)
            self._btn_accept_all.pack_forget()
            self._btn_export.pack_forget()
        else:
            self._btn_save_crops.pack_forget()
            # Accept All sits after the YOLO-view separator; Export is right-anchored
            self._btn_accept_all.pack(after=self._btn_yolo_mode, side=tk.LEFT, padx=2, pady=3)
            self._btn_export.pack(side=tk.RIGHT, padx=4, pady=3)

    def toggle_yolo_view_mode(self) -> None:
        """Cycle the display format of YOLO labels (box, point, both, or none)."""
        if self._yolo_view_mode == "box":
            self._yolo_view_mode = "point"
        elif self._yolo_view_mode == "point":
            self._yolo_view_mode = "both"
        elif self._yolo_view_mode == "both":
            self._yolo_view_mode = "none"
        else:
            self._yolo_view_mode = "box"

        mode_name = self._yolo_view_mode.capitalize()
        self._btn_yolo_mode.config(text=f"YOLO: {mode_name} (Y)")

        self._redraw()
        self._update_status()

    def toggle_save_crops_mode(self) -> None:
        """Toggle whether PNG crops are saved to disk."""
        self._save_crops_mode = not self._save_crops_mode
        state = "On" if self._save_crops_mode else "Off"
        self._btn_save_crops.config(
            text=f"Crops: {state} (C)", relief=tk.SUNKEN if self._save_crops_mode else tk.RAISED
        )
        self._update_status()

    def _find_crop_at(
        self, canvas_x: float, canvas_y: float
    ) -> tuple[Optional[str], Optional[int]]:
        """Return (type, index) of the topmost annotation whose canvas rect contains (canvas_x, canvas_y)."""
        # Check manual annotations first (they are on top)
        for i in range(len(self._manual_annots) - 1, -1, -1):
            x1, y1, x2, y2, _ = self._manual_annots[i]
            rect_canvas_x1, rect_canvas_y1 = self._image_to_canvas(x1, y1)
            rect_canvas_x2, rect_canvas_y2 = self._image_to_canvas(x2, y2)
            if (
                rect_canvas_x1 <= canvas_x <= rect_canvas_x2
                and rect_canvas_y1 <= canvas_y <= rect_canvas_y2
            ):
                return "manual", i

        # Check YOLO reference labels
        for i in range(len(self._yolo_labels) - 1, -1, -1):
            x1, y1, x2, y2, _ = self._yolo_labels[i]
            rect_canvas_x1, rect_canvas_y1 = self._image_to_canvas(x1, y1)
            rect_canvas_x2, rect_canvas_y2 = self._image_to_canvas(x2, y2)
            if (
                rect_canvas_x1 <= canvas_x <= rect_canvas_x2
                and rect_canvas_y1 <= canvas_y <= rect_canvas_y2
            ):
                return "yolo", i

        return None, None

    def _get_handle_at(self, canvas_x: float, canvas_y: float, idx: int) -> Optional[str]:
        """Return 'nw'|'ne'|'sw'|'se' if (canvas_x, canvas_y) is near a corner handle of crop[idx]."""
        x1, y1, x2, y2, _ = self._manual_annots[idx]
        rect_canvas_x1, rect_canvas_y1 = self._image_to_canvas(x1, y1)
        rect_canvas_x2, rect_canvas_y2 = self._image_to_canvas(x2, y2)
        handles = [
            ("nw", rect_canvas_x1, rect_canvas_y1),
            ("ne", rect_canvas_x2, rect_canvas_y1),
            ("sw", rect_canvas_x1, rect_canvas_y2),
            ("se", rect_canvas_x2, rect_canvas_y2),
        ]
        for name, handle_x, handle_y in handles:
            if abs(canvas_x - handle_x) <= HANDLE_HIT and abs(canvas_y - handle_y) <= HANDLE_HIT:
                return name
        return None

    def _on_edit_start(self, event: tk.Event) -> None:
        canvas_x, canvas_y = float(event.x), float(event.y)
        is_control = (event.state & 0x0004) != 0  # Control key held

        # 1. Handle resizing (only if exactly one manual crop was already selected)
        if len(self._selected_indices) == 1 and self._selected_type == "manual":
            idx = next(iter(self._selected_indices))
            handle = self._get_handle_at(canvas_x, canvas_y, idx)
            if handle:
                self._edit_drag_mode = f"resize_{handle}"
                self._edit_drag_start_canvas = (event.x, event.y)
                x1, y1, x2, y2, _ = self._manual_annots[idx]
                self._edit_drag_crops_orig = {idx: (x1, y1, x2, y2)}
                return

        # 2. Hit-test all crops (topmost first)
        kind, idx = self._find_crop_at(canvas_x, canvas_y)

        if idx is not None:
            if is_control and kind == "manual":
                # Toggle selection in multi-select mode
                if idx in self._selected_indices:
                    self._selected_indices.remove(idx)
                else:
                    self._selected_indices.add(idx)
                    self._selected_type = "manual"  # Multi-select only supported for manual
            else:
                # Normal selection: select only this one
                self._selected_indices = {idx}
                self._selected_type = kind

            # Setup drag if it's a manual crop
            if kind == "manual" and idx in self._selected_indices:
                self._btn_delete.config(state=tk.NORMAL)
                # Enable convert button if any selected don't have PNGs
                self._btn_convert.config(state=tk.NORMAL)  # simplified check

                handle = self._get_handle_at(canvas_x, canvas_y, idx)
                self._edit_drag_mode = (
                    f"resize_{handle}" if (handle and len(self._selected_indices) == 1) else "move"
                )
                self._edit_drag_start_canvas = (event.x, event.y)

                # Store original coords for all selected crops
                self._edit_drag_crops_orig = {}
                for s_idx in self._selected_indices:
                    x1, y1, x2, y2, _ = self._manual_annots[s_idx]
                    self._edit_drag_crops_orig[s_idx] = (x1, y1, x2, y2)
            else:
                # YOLO selection
                self._btn_delete.config(state=tk.DISABLED)
                self._btn_convert.config(state=tk.NORMAL)
                self._edit_drag_mode = None
        else:
            # Click on empty area
            if not is_control:
                # Deselect if control not held
                self._selected_indices = set()
                self._selected_type = "manual"
                self._edit_drag_mode = None
                self._edit_drag_start_canvas = None
                self._edit_drag_crops_orig = {}
                self._btn_delete.config(state=tk.DISABLED)
                self._btn_convert.config(state=tk.DISABLED)

        self._redraw()

    def _on_edit_drag(self, event: tk.Event) -> None:  # pylint: disable=too-many-locals
        if (
            self._edit_drag_mode is None
            or self._edit_drag_start_canvas is None
            or not self._selected_indices
            or not self._edit_drag_crops_orig
            or self.pil_image is None
        ):
            return

        display_scale = self.display_scale
        img_x_delta = (event.x - self._edit_drag_start_canvas[0]) / display_scale
        img_y_delta = (event.y - self._edit_drag_start_canvas[1]) / display_scale
        img_width, img_height = self.pil_image.size

        if self._edit_drag_mode == "move":
            for idx in self._selected_indices:
                orig_x1, orig_y1, orig_x2, orig_y2 = self._edit_drag_crops_orig[idx]
                _, _, _, _, path = self._manual_annots[idx]
                width, height = orig_x2 - orig_x1, orig_y2 - orig_y1
                new_x1 = int(max(0, min(img_width - width, orig_x1 + img_x_delta)))
                new_y1 = int(max(0, min(img_height - height, orig_y1 + img_y_delta)))
                new_x2 = new_x1 + width
                new_y2 = new_y1 + height
                self._manual_annots[idx] = (new_x1, new_y1, new_x2, new_y2, path)
        else:
            # Resizing only supported for a single selection (checked in _on_edit_start)
            idx = next(iter(self._selected_indices))
            orig_x1, orig_y1, orig_x2, orig_y2 = self._edit_drag_crops_orig[idx]
            _, _, _, _, path = self._manual_annots[idx]
            corner = self._edit_drag_mode[len("resize_") :]
            new_x1, new_y1, new_x2, new_y2 = orig_x1, orig_y1, orig_x2, orig_y2
            if "w" in corner:
                new_x1 = int(max(0, min(orig_x2 - MIN_CROP_PX, orig_x1 + img_x_delta)))
            if "e" in corner:
                new_x2 = int(max(orig_x1 + MIN_CROP_PX, min(img_width, orig_x2 + img_x_delta)))
            if "n" in corner:
                new_y1 = int(max(0, min(orig_y2 - MIN_CROP_PX, orig_y1 + img_y_delta)))
            if "s" in corner:
                new_y2 = int(max(orig_y1 + MIN_CROP_PX, min(img_height, orig_y2 + img_y_delta)))
            self._manual_annots[idx] = (new_x1, new_y1, new_x2, new_y2, path)

        self._redraw()

    def _on_edit_end(self, _event: tk.Event) -> None:  # pylint: disable=too-many-locals
        if (
            self._edit_drag_mode is None
            or not self._edit_drag_crops_orig
            or not self._selected_indices
            or self.pil_image is None
        ):
            self._edit_drag_mode = None
            self._edit_drag_crops_orig = {}
            return

        # Check if anything actually moved
        any_changed = False
        for idx in self._selected_indices:
            if idx not in self._edit_drag_crops_orig:
                continue
            if self._manual_annots[idx][:4] != self._edit_drag_crops_orig[idx]:
                any_changed = True
                break

        if not any_changed:
            self._edit_drag_mode = None
            self._edit_drag_start_canvas = None
            self._edit_drag_crops_orig = {}
            return

        try:
            for idx in self._selected_indices:
                new_x1, new_y1, new_x2, new_y2, old_path = self._manual_annots[idx]
                orig_x1, orig_y1, orig_x2, orig_y2 = self._edit_drag_crops_orig[idx]

                new_path = None
                if self._save_crops_mode:
                    if self.crops_dir is None:
                        self._status.set("No crops directory")
                    else:
                        stem = self._current_stem()
                        base = f"crop_{stem}_y{new_y1:04d}_x{new_x1:04d}"
                        new_path = self.crops_dir / f"{base}.png"
                        count = 1
                        while new_path.exists() and new_path != old_path:
                            new_path = self.crops_dir / f"{base}_{count}.png"
                            count += 1
                        try:
                            crop_img = self.pil_image.crop((new_x1, new_y1, new_x2, new_y2))
                            crop_img.save(new_path)
                            if old_path != new_path and old_path:
                                old_path.unlink(missing_ok=True)
                        except Exception as exc:  # pylint: disable=broad-except
                            self._status.set(f"Crop save failed: {exc}")
                else:
                    if old_path:
                        old_path.unlink(missing_ok=True)

                self._manual_annots[idx] = (new_x1, new_y1, new_x2, new_y2, new_path)

            self._sync_yolo_labels_file()
            self._undo_stack.append(
                {
                    "type": "multi_edit",
                    "orig_states": self._edit_drag_crops_orig.copy(),
                    "new_states": {
                        idx: self._manual_annots[idx][:4] for idx in self._selected_indices
                    },
                }
            )
            self._edit_drag_mode = None
            self._edit_drag_start_canvas = None
            self._edit_drag_crops_orig = {}
            self._redraw()
            self._update_status()
        except Exception as exc:  # pylint: disable=broad-except
            self._status.set(f"Save failed: {exc}")
            # Revert
            for idx, coords in self._edit_drag_crops_orig.items():
                old_path = self._manual_annots[idx][4]
                self._manual_annots[idx] = coords + (old_path,)
            self._edit_drag_mode = None
            self._edit_drag_crops_orig = {}
            self._redraw()

    # ── Folder / image management ─────────────────────────────────────────────

    def open_folder(self) -> None:
        """Prompt the user to select an image folder."""
        path = filedialog.askdirectory(
            title="Select image folder", initialdir=str(self.folder or Path.home())
        )
        if path:
            self._open_folder_path(Path(path))

    def _open_folder_path(self, folder: Path) -> None:
        self.folder = folder
        self._status.set(f"Scanning directory {folder.name}...")
        self.progress_bar.config(mode="determinate", value=0, maximum=100)
        self.progress_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.root.update_idletasks()
        self.root.update()

        def scan_job() -> None:
            paths = []
            try:
                names = os.listdir(folder)
                total = len(names)
                step = max(1, total // 100)
                self.root.after(0, lambda: self.progress_bar.config(maximum=total))

                for i, name in enumerate(names):
                    if i % step == 0:
                        self.root.after(0, lambda pos=i: self.progress_bar.config(value=pos))

                    _, ext = os.path.splitext(name)
                    if ext.lower() in IMAGE_EXTS:
                        img_path = folder / name
                        if img_path.is_file():
                            paths.append(img_path)
            except Exception as e:  # pylint: disable=broad-except
                self.root.after(0, lambda: _finish_scan(None, e))
                return
            self.root.after(0, lambda: _finish_scan(sorted(paths), None))

        def _finish_scan(paths: Optional[list[Path]], error: Optional[Exception]) -> None:
            self.progress_bar.stop()
            self.progress_bar.pack_forget()

            if error:
                messagebox.showerror("Error", f"Failed to open folder:\n{error}")
                self._status.set("Error opening folder.")
                return

            if paths is not None:
                self.image_paths = paths
            else:
                self.image_paths = []

            if not self.image_paths:
                messagebox.showinfo("No images", f"No supported images found in:\n{folder}")
                self._status.set("No images loaded.")
                return

            self._ensure_crops_dir()
            self.load_image(0)

        threading.Thread(target=scan_job, daemon=True).start()

    @property
    def crops_dir(self) -> Optional[Path]:
        """Return the path to the crops subdirectory if a folder is loaded."""
        return (self.folder / CROPS_SUBDIR) if self.folder else None

    def _ensure_crops_dir(self) -> Optional[Path]:
        if self.folder is None:
            return None
        crops_directory = self.folder / CROPS_SUBDIR
        crops_directory.mkdir(exist_ok=True)
        return crops_directory

    @property
    def labels_dir(self) -> Optional[Path]:
        """Path to labels/ folder, prefering adjacent to images/ then within it."""
        if self.folder is None:
            return None
        adj = self.folder.parent / "labels"
        if adj.exists() and adj.is_dir():
            return adj
        inner = self.folder / "labels"
        return inner

    def _ensure_labels_dir(self) -> Optional[Path]:
        if self.folder is None:
            return None
        labels_directory = self.labels_dir
        if labels_directory is None or not labels_directory.exists():
            # If folder is called 'images', create 'labels' next to it
            if self.folder.name == "images":
                labels_directory = self.folder.parent / "labels"
            else:
                labels_directory = self.folder / "labels"
            labels_directory.mkdir(parents=True, exist_ok=True)
        return labels_directory

    def prev_image(self) -> None:
        """Navigate to the previous image in the directory."""
        if self.current_index > 0:
            self.load_image(self.current_index - 1)

    def next_image(self) -> None:
        """Navigate to the next image in the directory."""
        if self.current_index < len(self.image_paths) - 1:
            self.load_image(self.current_index + 1)

    def _get_image(self, path: Path) -> Image.Image:
        """Load and normalize image, using cache if available."""
        if path in self._image_cache:
            return self._image_cache[path]

        img = Image.open(path)
        # Handle multi-frame images (TIF stacks): always use frame 0
        try:
            img.seek(0)
        except (EOFError, AttributeError):
            pass
        img = img.copy()  # detach from the file handle

        # Normalize exotic bit-depths to uint8
        if img.mode not in ("RGB", "RGBA", "L", "P"):
            arr = np.array(img).astype(float)
            arr_ptp = arr.max() - arr.min()
            arr = ((arr - arr.min()) / (arr_ptp if arr_ptp else 1) * 255).astype("uint8")
            img = Image.fromarray(arr)

        # Unify to RGB
        if img.mode == "P":
            img = img.convert("RGBA")
        if img.mode == "RGBA":
            background = Image.new("RGB", img.size, (30, 30, 30))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode == "L":
            img = img.convert("RGB")

        # Manage cache size (LRU-ish)
        if len(self._image_cache) >= self._cache_size:
            # Pop the first key (oldest)
            self._image_cache.pop(next(iter(self._image_cache)))

        self._image_cache[path] = img
        return img

    def _preload_neighbors(self) -> None:
        """Load next/prev images in a background thread."""

        def _target():
            indices = []
            if self.current_index + 1 < len(self.image_paths):
                indices.append(self.current_index + 1)
            if self.current_index - 1 >= 0:
                indices.append(self.current_index - 1)

            for idx in indices:
                path = self.image_paths[idx]
                if path not in self._image_cache:
                    try:
                        self._get_image(path)
                    except Exception:  # pylint: disable=broad-except
                        pass

        threading.Thread(target=_target, daemon=True).start()

    def load_image(self, index: int) -> None:
        """Load the image at the given index and update UI state."""
        self.current_index = index
        path = self.image_paths[index]

        try:
            self.pil_image = self._get_image(path)
            self.cv_image = np.array(self.pil_image)
        except Exception as e:  # pylint: disable=broad-except
            self._status.set(f"Error loading {path.name}: {e}")
            return

        # Load annotations before the first redraw so overlays appear immediately
        self._load_manual_annotations()
        self._load_yolo_labels()
        self._undo_stack.clear()

        # Reset edit selection on navigation (keep edit mode active for convenience)
        self._selected_indices = set()
        self._selected_type = "manual"
        self._edit_drag_mode = None
        self._edit_drag_start_canvas = None
        self._edit_drag_crops_orig = {}
        self._btn_delete.config(state=tk.DISABLED)
        self._btn_convert.config(state=tk.DISABLED)

        # Reset pan when navigating (intentionally preserve zoom level so
        self.pan_x = 0.0
        self.pan_y = 0.0

        # Redraw
        self._fit_to_canvas()

        # Update toolbar
        total_images = len(self.image_paths)
        self._lbl_counter.config(text=f"{index + 1} / {total_images}")
        self._btn_prev.config(state=tk.NORMAL if index > 0 else tk.DISABLED)
        self._btn_next.config(state=tk.NORMAL if index < total_images - 1 else tk.DISABLED)
        self.root.title(f"Crop Tool — {path.name}")

        self._update_status()
        self._preload_neighbors()

    # ── Transform helpers ─────────────────────────────────────────────────────

    @property
    def display_scale(self) -> float:
        """Pixels-per-image-pixel on the canvas."""
        return self.fit_scale * self.zoom_level

    def _fit_to_canvas(self) -> None:
        """Compute fit_scale and letterbox offsets; defer if canvas not yet sized."""
        if self.pil_image is None:
            return
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            self.root.after(50, self._fit_to_canvas)
            return
        img_width, img_height = self.pil_image.size
        self.fit_scale = min(canvas_width / img_width, canvas_height / img_height)
        self.img_offset_x = (canvas_width - img_width * self.fit_scale) / 2
        self.img_offset_y = (canvas_height - img_height * self.fit_scale) / 2
        self._redraw()

    def _canvas_to_image(self, canvas_x: float, canvas_y: float) -> tuple[float, float]:
        display_scale = self.display_scale
        return (
            (canvas_x - self.img_offset_x - self.pan_x) / display_scale,
            (canvas_y - self.img_offset_y - self.pan_y) / display_scale,
        )

    def _image_to_canvas(self, img_x: float, img_y: float) -> tuple[float, float]:
        display_scale = self.display_scale
        return (
            img_x * display_scale + self.img_offset_x + self.pan_x,
            img_y * display_scale + self.img_offset_y + self.pan_y,
        )

    def _clamp_to_image(self, img_x: float, img_y: float) -> tuple[int, int]:
        if self.pil_image is None:
            return 0, 0
        img_width, img_height = self.pil_image.size
        return int(max(0, min(img_width, img_x))), int(max(0, min(img_height, img_y)))

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _redraw(self) -> None:  # pylint: disable=too-many-locals
        if self.pil_image is None or self.cv_image is None:
            self._draw_welcome_screen()
            return
        display_scale = self.display_scale
        img_width, img_height = self.pil_image.size
        new_width = max(1, int(img_width * display_scale))
        new_height = max(1, int(img_height * display_scale))

        self.canvas.delete("all")
        self._cross_v_id = None
        self._cross_h_id = None

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 800, 600

        img_x = int(self.img_offset_x + self.pan_x)
        img_y = int(self.img_offset_y + self.pan_y)

        scaled_x_start = max(0, -img_x)
        scaled_y_start = max(0, -img_y)
        scaled_x_end = min(new_width, canvas_width - img_x)
        scaled_y_end = min(new_height, canvas_height - img_y)

        origin_x_start = int(scaled_x_start / display_scale)
        origin_y_start = int(scaled_y_start / display_scale)
        origin_x_end = int(math.ceil(scaled_x_end / display_scale))
        origin_y_end = int(math.ceil(scaled_y_end / display_scale))

        origin_x_start = max(0, min(img_width - 1, origin_x_start))
        origin_y_start = max(0, min(img_height - 1, origin_y_start))
        origin_x_end = max(0, min(img_width, origin_x_end))
        origin_y_end = max(0, min(img_height, origin_y_end))

        if origin_x_end <= origin_x_start or origin_y_end <= origin_y_start:
            self._draw_crop_overlays()
            return

        crop = self.cv_image[origin_y_start:origin_y_end, origin_x_start:origin_x_end]

        crop_new_width = max(1, int((origin_x_end - origin_x_start) * display_scale))
        crop_new_height = max(1, int((origin_y_end - origin_y_start) * display_scale))

        # pylint: disable=no-member
        interp = cv2.INTER_NEAREST if display_scale > 2 else cv2.INTER_LINEAR
        resized_crop = cv2.resize(crop, (crop_new_width, crop_new_height), interpolation=interp)
        # pylint: enable=no-member
        self._tk_img = ImageTk.PhotoImage(image=Image.fromarray(resized_crop))

        anchor_x = img_x + int(origin_x_start * display_scale)
        anchor_y = img_y + int(origin_y_start * display_scale)

        self._img_id = self.canvas.create_image(
            anchor_x, anchor_y, anchor=tk.NW, image=self._tk_img, tags="image"
        )
        self._draw_crop_overlays()
        self._update_crosshair()
        self._draw_color_legend()

    def export_yolo_dataset(self) -> None:
        """Export all images and their labels into a standard YOLOv8 structure."""
        if not self.folder or not self.image_paths:
            messagebox.showwarning("Export", "No images loaded to export.")
            return

        # 1. Ask for export destination
        dest = filedialog.askdirectory(title="Select Export Destination")
        if not dest:
            return

        export_path = Path(dest) / "yolo_dataset"
        img_dir = export_path / "images"
        lbl_dir = export_path / "labels"

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        # 2. Count images with labels
        exported_count = 0

        # Scan all images in the folder
        for img_path in self.image_paths:
            stem = img_path.stem
            # We assume labels are in our standard labels/ directory
            src_lbl = self.labels_dir / f"{stem}.txt" if self.labels_dir else None

            if src_lbl and src_lbl.exists():
                # Copy image
                shutil.copy2(img_path, img_dir / img_path.name)
                # Copy label
                shutil.copy2(src_lbl, lbl_dir / f"{stem}.txt")
                exported_count += 1

        if exported_count == 0:
            messagebox.showinfo("Export", "No labels found to export. Start labeling first!")
            return

        # 3. Create data.yaml
        yaml_content = f"""path: {export_path.absolute()}
train: images
val: images

names:
  0: particle
"""
        with open(export_path / "data.yaml", "w", encoding="utf-8") as f:
            f.write(yaml_content)

        messagebox.showinfo(
            "Export Successful",
            f"Exported {exported_count} labeled images to:\n{export_path}\n\nReady for YOLO training!",
        )
        self._status.set(f"Dataset exported to {export_path.name}")

    def _draw_welcome_screen(self) -> None:
        """Show instructions on a blank canvas."""
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 1:
            w = 800
        if h <= 1:
            h = 600

        self.canvas.create_text(
            w / 2,
            h / 2 - 60,
            text="WELCOME TO LodeSTAR CROP TOOL",
            fill="white",
            font=("Helvetica", 18, "bold"),
            tags="welcome",
        )

        instructions = [
            "1. Click 'Open Folder' to load your frames.",
            "2. LEFT-CLICK & DRAG to draw a crop box.",
            "3. RIGHT-CLICK & DRAG to pan the image.",
            "4. MOUSE WHEEL to zoom in/out.",
            "",
            "Tip: Press 'C' to enable automatic PNG saving!",
            "Press '?' for all keyboard shortcuts.",
        ]

        for i, line in enumerate(instructions):
            self.canvas.create_text(
                w / 2,
                h / 2 + (i * 25),
                text=line,
                fill="#cccccc",
                font=("Helvetica", 11),
                tags="welcome",
            )

    def _draw_color_legend(self) -> None:
        """Draw a small color legend in the corner."""
        self.canvas.create_rectangle(
            5, 5, 160, 75, fill="#1e1e1e", outline="#444444", stipple="gray50", tags="legend"
        )

        items = [("#00FF88", "Manual Crop"), ("#FF00FF", "Auto Detection"), ("#FFD700", "Selected")]

        for i, (color, label) in enumerate(items):
            # Symbol
            self.canvas.create_rectangle(
                15, 15 + i * 20, 25, 25 + i * 20, fill=color, outline="white", tags="legend"
            )
            # Text
            self.canvas.create_text(
                35,
                20 + i * 20,
                text=label,
                fill="white",
                anchor=tk.W,
                font=("Helvetica", 9),
                tags="legend",
            )

    def _draw_crop_overlays(self) -> None:  # pylint: disable=too-many-locals
        """Draw a rectangle on the canvas for each manual annotation."""
        for i, (x1, y1, x2, y2, _) in enumerate(self._manual_annots):
            cx1, cy1 = self._image_to_canvas(x1, y1)
            cx2, cy2 = self._image_to_canvas(x2, y2)
            selected = (
                self._edit_mode and self._selected_type == "manual" and i in self._selected_indices
            )
            color = "#FFD700" if selected else "#00FF88"
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline=color, width=2, tags="overlay")
            if selected:
                for handle_x, handle_y in ((cx1, cy1), (cx2, cy1), (cx1, cy2), (cx2, cy2)):
                    radius = HANDLE_RADIUS
                    self.canvas.create_rectangle(
                        handle_x - radius,
                        handle_y - radius,
                        handle_x + radius,
                        handle_y + radius,
                        fill="#FFD700",
                        outline="#FF8C00",
                        width=1,
                        tags="overlay",
                    )

        # Draw YOLO labels (fixed detections)
        if self._yolo_view_mode != "none":
            for i, (x1, y1, x2, y2, cls_id) in enumerate(self._yolo_labels):
                cx1, cy1 = self._image_to_canvas(x1, y1)
                cx2, cy2 = self._image_to_canvas(x2, y2)

                selected = (
                    self._edit_mode
                    and self._selected_type == "yolo"
                    and i in self._selected_indices
                )
                tag_color = "#00FFFF" if selected else "#FF00FF"  # Cyan if selected, else Magenta

                # Draw YOLO labels based on mode
                show_box = self._yolo_view_mode in ("box", "both")
                show_point = self._yolo_view_mode in ("point", "both")

                if show_box:
                    self.canvas.create_rectangle(
                        cx1,
                        cy1,
                        cx2,
                        cy2,
                        fill="",
                        outline=tag_color,
                        width=2 if selected else 1,
                        dash=None if selected else (2, 2),
                        tags="overlay",
                    )

                if show_point:
                    mid_x, mid_y = (cx1 + cx2) / 2, (cy1 + cy2) / 2
                    radius = 4 if selected else 3
                    self.canvas.create_oval(
                        mid_x - radius,
                        mid_y - radius,
                        mid_x + radius,
                        mid_y + radius,
                        fill=tag_color,
                        outline="white",
                        width=1,
                        tags="overlay",
                    )

                # Label the class
                label_x = cx1 if show_box else (cx1 + cx2) / 2
                label_y = (cy1 - 2) if show_box else (cy1 + cy2) / 2 - 5
                label_anchor = tk.SW if show_box else tk.S

                self.canvas.create_text(
                    label_x,
                    label_y,
                    text=f"cls:{cls_id}",
                    fill=tag_color,
                    anchor=label_anchor,
                    font=("Helvetica", 8, "bold"),
                    tags="overlay",
                )

    # ── Zoom & pan ────────────────────────────────────────────────────────────

    def zoom_in(self) -> None:
        """Zoom into the image."""
        canvas_width, canvas_height = self.canvas.winfo_width(), self.canvas.winfo_height()
        self._zoom_at(canvas_width / 2, canvas_height / 2, ZOOM_STEP)

    def zoom_out(self) -> None:
        """Zoom out of the image."""
        canvas_width, canvas_height = self.canvas.winfo_width(), self.canvas.winfo_height()
        self._zoom_at(canvas_width / 2, canvas_height / 2, 1 / ZOOM_STEP)

    def _zoom_at(self, canvas_x: float, canvas_y: float, factor: float) -> None:
        """Zoom by factor, keeping the canvas point (canvas_x, canvas_y) fixed over the same image pixel."""
        new_zoom = max(ZOOM_MIN, min(ZOOM_MAX, self.zoom_level * factor))
        if abs(new_zoom - self.zoom_level) < 1e-9:
            return
        # image point under cursor must stay at the same canvas position after zoom:
        # canvas_x = ix * new_ds + offset_x + new_pan_x  =>  new_pan_x = cx - offset_x - ix * new_ds
        img_x, img_y = self._canvas_to_image(canvas_x, canvas_y)
        self.zoom_level = new_zoom
        new_display_scale = self.display_scale
        self.pan_x = canvas_x - self.img_offset_x - img_x * new_display_scale
        self.pan_y = canvas_y - self.img_offset_y - img_y * new_display_scale
        self._redraw()
        self._update_status()

    def reset_transform(self) -> None:
        """Reset zoom and pan to fit the image to the canvas."""
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._fit_to_canvas()
        self._update_status()

    def _on_mousewheel(self, event: tk.Event) -> None:
        if self.pil_image is None:
            return
        if event.num == 4 or (hasattr(event, "delta") and event.delta > 0):
            self._zoom_at(event.x, event.y, ZOOM_STEP)
        else:
            self._zoom_at(event.x, event.y, 1 / ZOOM_STEP)

    def _on_pan_start(self, event: tk.Event) -> None:
        self._pan_start_canvas = (event.x, event.y)
        self._pan_start_offset = (self.pan_x, self.pan_y)
        self.canvas.config(cursor="fleur")

    def _on_pan_drag(self, event: tk.Event) -> None:
        self._on_mouse_move(event)
        if self._pan_start_canvas is None:
            return
        delta_x = event.x - self._pan_start_canvas[0]
        delta_y = event.y - self._pan_start_canvas[1]

        # Real-time shift of all canvas items (don't redraw yet)
        shift_x = delta_x - (self.pan_x - self._pan_start_offset[0])
        shift_y = delta_y - (self.pan_y - self._pan_start_offset[1])

        self.pan_x = self._pan_start_offset[0] + delta_x
        self.pan_y = self._pan_start_offset[1] + delta_y

        self.canvas.move("all", shift_x, shift_y)

    def _on_pan_end(self, _event: tk.Event) -> None:
        self._pan_start_canvas = None
        self.canvas.config(cursor="crosshair")

    # ── Rubber-band crop ──────────────────────────────────────────────────────

    def _on_mouse_move(self, event: tk.Event) -> None:
        """Update software crosshair position."""
        self._mouse_pos = (event.x, event.y)
        self._update_crosshair()

    def _on_mouse_leave(self, _event: tk.Event) -> None:
        """Hide crosshair when mouse leaves canvas."""
        self._mouse_pos = None
        if self._cross_v_id:
            self.canvas.itemconfigure(self._cross_v_id, state=tk.HIDDEN)
        if self._cross_h_id:
            self.canvas.itemconfigure(self._cross_h_id, state=tk.HIDDEN)

    def _update_crosshair(self) -> None:
        """Draw or move the software crosshair lines."""
        if self._mouse_pos is None:
            return

        x, y = self._mouse_pos
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        # Create lines if they don't exist
        if self._cross_v_id is None:
            self._cross_v_id = self.canvas.create_line(
                x, 0, x, h, fill="#ff4d4d", dash=(4, 4), width=1, state=tk.NORMAL, tags="crosshair"
            )
            self._cross_h_id = self.canvas.create_line(
                0, y, w, y, fill="#ff4d4d", dash=(4, 4), width=1, state=tk.NORMAL, tags="crosshair"
            )
        else:
            self.canvas.coords(self._cross_v_id, x, 0, x, h)
            self.canvas.coords(self._cross_h_id, 0, y, w, y)
            self.canvas.itemconfigure(self._cross_v_id, state=tk.NORMAL)
            self.canvas.itemconfigure(self._cross_h_id, state=tk.NORMAL)
            # Ensure they stay on top
            self.canvas.tag_raise("crosshair")

        # Also update preview box if in fixed mode and not dragging
        if self._fixed_size and self._drag_start is None and not self._edit_mode:
            self._update_fixed_preview()

    def _update_fixed_preview(self) -> None:
        """Draw a preview box of the fixed size centered on the mouse."""
        if self._mouse_pos is None or self._fixed_size is None:
            return

        x, y = self._mouse_pos
        ds = self.display_scale
        half = (self._fixed_size * ds) / 2

        # Draw preview rectangle
        if not hasattr(self, "_fixed_preview_id") or self._fixed_preview_id is None:
            self._fixed_preview_id = self.canvas.create_rectangle(
                x - half,
                y - half,
                x + half,
                y + half,
                outline="#00FF88",
                dash=(2, 2),
                tags="fixed_preview",
            )
        else:
            self.canvas.coords(self._fixed_preview_id, x - half, y - half, x + half, y + half)
            self.canvas.itemconfigure(self._fixed_preview_id, state=tk.NORMAL)

        self.canvas.tag_raise("fixed_preview")

    def toggle_fixed_size(self) -> None:
        """Toggle or set the fixed size for cropping."""
        if self._fixed_size:
            self._fixed_size = None
            self._btn_fixed_size.config(text="Fixed Size: Off (F)", relief=tk.RAISED)
            if hasattr(self, "_fixed_preview_id") and self._fixed_preview_id:
                self.canvas.itemconfigure(self._fixed_preview_id, state=tk.HIDDEN)
        else:
            size_str = tk.simpledialog.askstring(
                "Fixed Size", "Enter square size in pixels (e.g., 64):", initialvalue="64"
            )
            if size_str and size_str.isdigit():
                self._fixed_size = int(size_str)
                self._btn_fixed_size.config(
                    text=f"Fixed: {self._fixed_size}px (F)", relief=tk.SUNKEN
                )
            else:
                self._status.set("Invalid size.")
        self._update_status()

    def _on_drag_start(self, event: tk.Event) -> None:
        if self.pil_image is None:
            return
        if self._edit_mode:
            self._on_edit_start(event)
            return
        self._drag_start = (event.x, event.y)
        self._rect_id = self.canvas.create_rectangle(
            event.x,
            event.y,
            event.x,
            event.y,
            outline="red",
            width=2,
            dash=(6, 4),
            tags="rubberband",
        )
        # Create internal crosshair for selection box
        self._rect_cross_v_id = self.canvas.create_line(
            event.x, event.y, event.x, event.y, fill="#00FF88", width=1, tags="rubberband"
        )
        self._rect_cross_h_id = self.canvas.create_line(
            event.x, event.y, event.x, event.y, fill="#00FF88", width=1, tags="rubberband"
        )

    def _on_drag_motion(self, event: tk.Event) -> None:
        self._on_mouse_move(event)
        if self._edit_mode:
            self._on_edit_drag(event)
            return
        if self._drag_start is None or self._rect_id is None:
            return
        x0, y0 = self._drag_start
        x1, y1 = event.x, event.y

        # Force Square
        dx = x1 - x0
        dy = y1 - y0
        side = max(abs(dx), abs(dy))
        x1 = x0 + (side if dx > 0 else -side)
        y1 = y0 + (side if dy > 0 else -side)

        # Update main rectangle
        self.canvas.coords(self._rect_id, x0, y0, x1, y1)

        # Update internal crosshairs
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        self.canvas.coords(self._rect_cross_v_id, mid_x, y0, mid_x, y1)
        self.canvas.coords(self._rect_cross_h_id, x0, mid_y, x1, mid_y)

    def _on_drag_end(self, event: tk.Event) -> None:  # pylint: disable=too-many-locals
        if self._edit_mode:
            self._on_edit_end(event)
            return
        if self._drag_start is None or self.pil_image is None:
            return
        x0c, y0c = self._drag_start
        self._drag_start = None
        if self._rect_id is not None:
            self.canvas.delete(self._rect_id)
            self._rect_id = None

        # Convert canvas coords to image coords, then clamp to image bounds
        img_x0_raw, img_y0_raw = self._canvas_to_image(float(x0c), float(y0c))
        img_x1_raw, img_y1_raw = self._canvas_to_image(float(event.x), float(event.y))
        img_x0, img_y0 = self._clamp_to_image(img_x0_raw, img_y0_raw)
        img_x1, img_y1 = self._clamp_to_image(img_x1_raw, img_y1_raw)

        # If in fixed mode, center the fixed box on the click start (or end)
        if self._fixed_size:
            side = self._fixed_size
            x1 = int(img_x0 - side // 2)
            y1 = int(img_y0 - side // 2)
            x2 = x1 + side
            y2 = y1 + side
        else:
            # Force square normalization based on the drag
            dx = img_x1 - img_x0
            dy = img_y1 - img_y0
            side = int(max(abs(dx), abs(dy)))
            x1 = img_x0 if dx > 0 else img_x0 - side
            y1 = img_y0 if dy > 0 else img_y0 - side
            x2 = x1 + side
            y2 = y1 + side

        # Final bounds clamp
        img_width, img_height = self.pil_image.size
        x1 = max(0, min(img_width - 1, x1))
        y1 = max(0, min(img_height - 1, y1))
        x2 = max(0, min(img_width, x2))
        y2 = max(0, min(img_height, y2))

        if (x2 - x1) < MIN_CROP_PX or (y2 - y1) < MIN_CROP_PX:
            return  # too small — ignore accidental clicks

        if self.pil_image is None or self.crops_dir is None:
            return

        new_path = None
        if self._save_crops_mode:
            stem = self._current_stem()
            base = f"crop_{stem}_y{y1:04d}_x{x1:04d}"
            save_path = self.crops_dir / f"{base}.png"
            count = 1
            while save_path.exists():
                save_path = self.crops_dir / f"{base}_{count}.png"
                count += 1

            try:
                crop_img = self.pil_image.crop((x1, y1, x2, y2))
                crop_img.save(save_path)
                new_path = save_path
            except Exception as e:  # pylint: disable=broad-except
                self._status.set(f"Crop save failed: {e}")

        self._manual_annots.append((x1, y1, x2, y2, new_path))
        self._sync_yolo_labels_file()
        self._undo_stack.append({"type": "create", "path": new_path})
        self._redraw()
        self._update_status()

    # ── Crop persistence ──────────────────────────────────────────────────────

    def _current_stem(self) -> str:
        return self.image_paths[self.current_index].stem if self.current_index >= 0 else ""

    def _load_manual_annotations(self) -> None:  # pylint: disable=too-many-locals
        """Load annotations from YOLO labels file and link to existing PNG crops."""
        self._manual_annots = []
        if self.folder is None or self.pil_image is None:
            return

        lbl_dir = self.labels_dir
        if not lbl_dir:
            return

        stem = self._current_stem()
        lbl_path = lbl_dir / f"{stem}.txt"
        if not lbl_path.exists():
            return

        img_width, img_height = self.pil_image.size
        try:
            with open(lbl_path, "r", encoding="utf-8") as label_file:
                for line in label_file:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    _cls_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])

                    px_width, px_height = width * img_width, height * img_height
                    px_x_center, px_y_center = x_center * img_width, y_center * img_height
                    x1, y1 = int(px_x_center - px_width / 2), int(px_y_center - px_height / 2)
                    x2, y2 = int(px_x_center + px_width / 2), int(px_y_center + px_height / 2)

                    # Try to find a matching PNG crop
                    match_path = None
                    if self.crops_dir and self.crops_dir.exists():
                        base_glob = f"crop_{stem}_y{y1:04d}_x{x1:04d}*.png"
                        matches = list(self.crops_dir.glob(base_glob))
                        if matches:
                            match_path = matches[0]

                    self._manual_annots.append((x1, y1, x2, y2, match_path))
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error loading manual annotations: {e}")

    def _load_existing_crops(self) -> None:
        """DEPRECATED: Replaced by _load_manual_annotations which syncs from labels."""

    def _load_yolo_labels(self) -> None:  # pylint: disable=too-many-locals
        """Scan for YOLO .txt labels in adjacent folders."""
        self._yolo_labels = []
        if self.folder is None or self.pil_image is None:
            return

        # 1. Try ../labels/ (if current is .../images/)
        # 2. Try ./labels/
        candidates = [self.folder.parent / "labels", self.folder / "labels"]

        lbl_dir = None
        for cand in candidates:
            if cand.exists() and cand.is_dir():
                lbl_dir = cand
                break

        if lbl_dir is None:
            return

        stem = self._current_stem()
        lbl_path = lbl_dir / f"{stem}.txt"

        if not lbl_path.exists():
            return

        img_width, img_height = self.pil_image.size
        try:
            with open(lbl_path, "r", encoding="utf-8") as label_file:
                for line in label_file:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])

                    # Convert normalized to pixel coordinates
                    px_width = width * img_width
                    px_height = height * img_height
                    px_x_center = x_center * img_width
                    px_y_center = y_center * img_height

                    x1 = int(px_x_center - px_width / 2)
                    y1 = int(px_y_center - px_height / 2)
                    x2 = int(px_x_center + px_width / 2)
                    y2 = int(px_y_center + px_height / 2)

                    self._yolo_labels.append((x1, y1, x2, y2, cls_id))
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error loading YOLO labels from {lbl_path}: {e}")

    def undo_last_crop(self) -> None:
        """Revert the last create, edit, or delete action."""
        if not self._undo_stack:
            self._status.set("Nothing to undo.")
            return
        entry = self._undo_stack.pop()
        kind = entry["type"]

        if kind == "create":
            if self._manual_annots:
                _, _, _, _, path = self._manual_annots.pop()
                if path:
                    path.unlink(missing_ok=True)
        elif kind == "delete":
            self._manual_annots.append((*entry["coords"], entry["path"]))
            # We don't restore the PNG file, just the logical annotation
        elif kind == "edit":
            # Search for the newly edited entry to revert it
            curr = (*entry["new_coords"], entry["new_path"])
            try:
                idx = self._manual_annots.index(curr)
                self._manual_annots[idx] = (*entry["old_coords"], entry["old_path"])
                if entry["new_path"]:
                    entry["new_path"].unlink(missing_ok=True)
            except ValueError:
                pass

        self._sync_yolo_labels_file()
        self._redraw()
        self._update_status()

    def _sync_yolo_labels_file(self) -> None:  # pylint: disable=too-many-locals
        """Write all current image's manual crops to the YOLO labels file."""
        lbl_dir = self._ensure_labels_dir()
        if not lbl_dir or not self.pil_image:
            return

        stem = self._current_stem()
        lbl_path = lbl_dir / f"{stem}.txt"

        if not self._manual_annots:
            if lbl_path.exists():
                lbl_path.unlink()
            return

        img_width, img_height = self.pil_image.size
        try:
            with open(lbl_path, "w", encoding="utf-8") as label_file:
                for x1, y1, x2, y2, _ in self._manual_annots:
                    # YOLO format: class x_center y_center width height (normalized)
                    width = x2 - x1
                    height = y2 - y1
                    x_center = x1 + width / 2
                    y_center = y1 + height / 2

                    n_xc = x_center / img_width
                    n_yc = y_center / img_height
                    n_w = width / img_width
                    n_h = height / img_height
                    label_file.write(f"0 {n_xc:.6f} {n_yc:.6f} {n_w:.6f} {n_h:.6f}\n")
        except Exception as e:  # pylint: disable=broad-except
            self._status.set(f"YOLO sync failed: {e}")

    def convert_yolo_to_manual(self, y_idx: int) -> bool:  # pylint: disable=too-many-locals
        """Convert a YOLO label to a manual annotation + PNG crop. Returns True if successful."""
        if self.pil_image is None or self.crops_dir is None:
            return False

        x1, y1, x2, y2, _ = self._yolo_labels[y_idx]

        # Avoid duplicate manual labels at the same coordinates
        for mx1, my1, mx2, my2, _ in self._manual_annots:
            if abs(mx1 - x1) < 2 and abs(my1 - y1) < 2 and abs(mx2 - x2) < 2 and abs(my2 - y2) < 2:
                return False

        stem = self._current_stem()
        base = f"crop_{stem}_y{y1:04d}_x{x1:04d}"
        save_path = self.crops_dir / f"{base}.png"
        count = 1
        while save_path.exists():
            save_path = self.crops_dir / f"{base}_{count}.png"
            count += 1

        try:
            crop_img = self.pil_image.crop((x1, y1, x2, y2))
            crop_img.save(save_path)
            self._manual_annots.append((x1, y1, x2, y2, save_path))
            return True
        except Exception as e:  # pylint: disable=broad-except
            print(f"Failed to convert YOLO to crop: {e}")
            return False

    def convert_selected(self) -> None:
        """Convert all selected YOLO labels or manual annotations to PNG crops."""
        if not self._selected_indices:
            self._status.set("Select a YOLO label (magenta) or a manual label (green) first.")
            return

        indices = sorted(list(self._selected_indices), reverse=True)
        new_selections = set()
        success_count = 0

        for idx in indices:
            if self._selected_type == "yolo":
                if self.convert_yolo_to_manual(idx):
                    new_selections.add(len(self._manual_annots) - 1)
                    success_count += 1
            else:
                # Manual crop conversion
                x1, y1, x2, y2, path = self._manual_annots[idx]
                if path is not None:
                    continue

                if self.crops_dir is None or self.pil_image is None:
                    continue

                stem = self._current_stem()
                base = f"crop_{stem}_y{y1:04d}_x{x1:04d}"
                save_path = self.crops_dir / f"{base}.png"
                count = 1
                while save_path.exists():
                    save_path = self.crops_dir / f"{base}_{count}.png"
                    count += 1

                try:
                    crop_img = self.pil_image.crop((x1, y1, x2, y2))
                    crop_img.save(save_path)
                    self._manual_annots[idx] = (x1, y1, x2, y2, save_path)
                    new_selections.add(idx)
                    success_count += 1
                except Exception as e:
                    self._status.set(f"Convert failed: {e}")

        if self._selected_type == "yolo" and new_selections:
            self._selected_type = "manual"
            self._selected_indices = new_selections

        if success_count > 0:
            self._sync_yolo_labels_file()
            self._redraw()
            self._update_status()
            self._status.set(f"Successfully converted {success_count} crops.")
        else:
            self._status.set("Conversion failed or crops already exist.")

    def convert_all_yolo(self) -> None:
        """Convert all YOLO labels in the current image to manual labels."""
        if not self._yolo_labels:
            self._status.set("No YOLO labels to convert.")
            return

        count = 0
        for i in range(len(self._yolo_labels)):
            if self.convert_yolo_to_manual(i):
                count += 1

        if count > 0:
            self._sync_yolo_labels_file()
            self._redraw()
            self._update_status()
            self._status.set(f"Converted {count} YOLO labels to LodeSTAR crops.")
        else:
            self._status.set("No new YOLO labels were converted.")

    def delete_selected_crop(self) -> None:
        """Delete all selected manual crops."""
        if not self._selected_indices or self._selected_type != "manual":
            return

        indices = sorted(list(self._selected_indices), reverse=True)
        deleted_count = 0

        for idx in indices:
            x1, y1, x2, y2, path = self._manual_annots[idx]
            # Store for undo
            self._undo_stack.append({"type": "delete", "path": path, "coords": (x1, y1, x2, y2)})

            if path:
                try:
                    path.unlink(missing_ok=True)
                except Exception as exc:  # pylint: disable=broad-except
                    print(f"Delete failed for {path}: {exc}")

            self._manual_annots.pop(idx)
            deleted_count += 1

        self._selected_indices = set()
        self._btn_delete.config(state=tk.DISABLED)
        self._sync_yolo_labels_file()
        self._redraw()
        self._update_status()
        self._status.set(f"Deleted {deleted_count} crops.")

    # ── Status bar ────────────────────────────────────────────────────────────

    def _update_status(self) -> None:
        if self.pil_image is None or self.current_index < 0:
            self._status.set("Open a folder to begin.")
            return
        name = self.image_paths[self.current_index].name
        img_width, img_height = self.pil_image.size
        zoom_pct = int(self.display_scale * 100)
        m_count = len(self._manual_annots)
        y_count = len(self._yolo_labels)
        mode = "EDIT" if self._edit_mode else "CROP"
        crops_gen = "Crops:ON" if self._save_crops_mode else "Crops:OFF"
        yolo_view = self._yolo_view_mode.upper()
        p_mode = self.project_mode.upper()

        self._status.set(
            f"[{p_mode}] {name} ({img_width}x{img_height}) | Zoom:{zoom_pct}% | {mode} | "
            f"{crops_gen} | YOLO:{yolo_view} | {m_count} labels | {y_count} ref"
        )

    # ── Canvas resize ─────────────────────────────────────────────────────────

    def show_help(self) -> None:
        """Display a help window with controls and legend."""
        help_win = tk.Toplevel(self.root)
        help_win.title("Crop Tool Help & Shortcuts")
        help_win.geometry("500x600")
        help_win.resizable(False, False)

        text = tk.Text(
            help_win, wrap=tk.WORD, font=("Helvetica", 10), padx=10, pady=10, bg="#f0f0f0"
        )
        text.pack(fill=tk.BOTH, expand=True)

        help_content = """
MANUAL CROP TOOL HELP
=====================

CORE MOUSE CONTROLS:
- Left-Click & Drag: Draw a new crop rectangle.
- Right-Click & Drag: Pan/move the image.
- Mouse Wheel: Zoom in/out at cursor position.

KEYBOARD SHORTCUTS:
- [Left / Right]: Previous / Next image.
- [R]: Reset zoom and pan to fit image.
- [E]: Toggle EDIT MODE (select and move boxes).
- [C]: Toggle CROP SAVING (save images to /crops folder).
- [Accept All Detections]: Turn all computer-found boxes on this frame into yours.
- [Export YOLO Dataset]: Create a 'yolo_dataset' folder with images, labels, and data.yaml.
- [T]: Convert selected detection to a LodeSTAR crop.
- [Ctrl + Z]: Undo last action.
- [Del / Backspace]: Delete selected crop (in Edit Mode).
- [+/-]: Zoom in / out.

COLOR LEGEND:
- GREEN BOXES: Your manual crops.
- MAGENTA BOXES/DOTS: Computer (YOLO) detections.
- YELLOW/CYAN: Currently selected items.

TIPS:
1. Turn 'Crops Mode: ON' to automatically save PNG files while drawing.
2. Use 'Edit Mode' to fix the size of boxes you've already drawn.
3. Training LodeSTAR? Use sharp frames and center the particles!
"""
        text.insert(tk.END, help_content)
        text.config(state=tk.DISABLED)

        tk.Button(help_win, text="Close", command=help_win.destroy).pack(pady=5)

    def _on_canvas_configure(self, event: tk.Event) -> None:
        if self.pil_image is None or event.width <= 1 or event.height <= 1:
            return
        if self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(
            40, lambda: self._do_resize(event.width, event.height)
        )

    def _do_resize(self, canvas_width: int, canvas_height: int) -> None:
        self._resize_after_id = None
        if self.pil_image is None:
            return
        img_width, img_height = self.pil_image.size
        new_fit = min(canvas_width / img_width, canvas_height / img_height)
        old_fit = self.fit_scale
        self.fit_scale = new_fit
        self.img_offset_x = (canvas_width - img_width * new_fit) / 2
        self.img_offset_y = (canvas_height - img_height * new_fit) / 2
        # Scale pan proportionally so the visible image region doesn't jump
        if old_fit > 0:
            ratio = new_fit / old_fit
            self.pan_x *= ratio
            self.pan_y *= ratio
        self._redraw()

    # ── Entry point ───────────────────────────────────────────────────────────

    def _parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Manual crop tool — browse images in a folder and draw/save crops.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=__doc__,
        )
        parser.add_argument("folder", nargs="?", help="Folder containing images to browse.")
        parser.add_argument("--config", "-c", help="Path to a JSON configuration file.")
        return parser.parse_args()

    def run(self) -> None:
        """Start the application main loop."""
        args = self._parse_args()
        target_folder: Optional[Path] = None

        # 1. Check if folder was provided as positional arg
        if args.folder:
            target_folder = Path(args.folder)

        # 2. Check config file if provided (overrides positional folder)
        if args.config:
            config_path = Path(args.config)
            if not config_path.exists():
                print(f"Error: Config file not found: {args.config}")
            else:
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config_data = json.load(f)

                    # If output_dir is in config, look for 'images' subfolder
                    if "output_dir" in config_data:
                        out_dir = Path(config_data["output_dir"])
                        img_dir = out_dir / "images"
                        if img_dir.is_dir():
                            target_folder = img_dir
                        elif out_dir.is_dir():
                            target_folder = out_dir

                    # Fallback to 'input' if it's a directory and target_folder not set
                    if not target_folder and "input" in config_data:
                        input_path = Path(config_data["input"])
                        if input_path.is_dir():
                            target_folder = input_path

                except Exception as e:
                    print(f"Error loading config file: {e}")

        # Auto-open the folder if valid
        if target_folder and target_folder.is_dir():
            self.root.after(500, lambda f=target_folder: self._open_folder_path(f))
        elif target_folder:
            print(f"Warning: {target_folder} is not a valid directory.")

        self.root.mainloop()


def _sep(parent: tk.Frame) -> None:
    """Insert a visual separator into a toolbar frame."""
    tk.Frame(parent, width=2, bd=1, relief=tk.SUNKEN).pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=2)


if __name__ == "__main__":
    CropTool().run()
