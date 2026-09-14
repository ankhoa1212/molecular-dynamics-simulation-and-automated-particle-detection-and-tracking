#!/bin/bash

# ==============================================================================
# A bash script to run a LAMMPS simulation using a filename provided as a
# command-line argument.

LAMMPS_EXECUTABLE="lmp"

# --- Set OpenMP Threads ---

# Automatically determine the number of available processor cores and set
# OMP_NUM_THREADS for multi-threaded execution with OpenMP packages.
if [[ -x "$(command -v nproc)" ]]; then
  # For Linux systems
  export OMP_NUM_THREADS=$(nproc)
else
  # Fallback if no command is found
  echo "Could not determine number of CPUs. Defaulting OMP_NUM_THREADS to 1."
  export OMP_NUM_THREADS=1
fi
echo "Setting OMP_NUM_THREADS to $OMP_NUM_THREADS"

OUTPUT_DIR="results"
MOLECULES="1000"
MOLECULES_END="$MOLECULES"
MOLECULES_STEP="1000"
VAR_EPSILON="5.0"
VAR_EPSILON_END="$VAR_EPSILON"
VAR_EPSILON_STEP="5.0"

# Validate argument count (the input script itself is required as $1)
if [ "$#" -lt 1 ] || [ "$#" -gt 8 ]; then
  echo "Usage: $0 <lammps_input_file> [output_directory] [molecules [molecules_end molecules_step]] [var_epsilon [var_epsilon_end var_epsilon_step]]"
  echo "Examples:"
  echo "  $0 central_pair_interaction.in"
  echo "  $0 central_pair_interaction.in results"
  echo "  $0 central_pair_interaction.in results 1000"
  echo "  $0 central_pair_interaction.in results 1000 2000 1000"
  echo "  $0 central_pair_interaction.in results 1000 2000 1000 5.0"
  echo "  $0 central_pair_interaction.in results 1000 2000 1000 5.0 10.0 5.0"
  exit 1
fi

# Second argument, if given, overrides OUTPUT_DIR
if [ "$#" -gt 1 ]; then
    OUTPUT_DIR="$2"
fi

# Third through fifth arguments, if given, override the molecules sweep
if [ "$#" -gt 2 ]; then
    MOLECULES="$3"
    MOLECULES_END="$4"
    MOLECULES_STEP="$5"
fi

# Sixth through eighth arguments, if given, override the epsilon sweep
if [ "$#" -gt 5 ]; then
    VAR_EPSILON="$6"
    VAR_EPSILON_END="$7"
    VAR_EPSILON_STEP="$8"
fi

INPUT_SCRIPT="$1"

mkdir -p "$OUTPUT_DIR"

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

# --- Run Simulation ---
echo "=========================================="
echo "Starting LAMMPS simulation..."
echo "  Executable: $LAMMPS_EXECUTABLE"
echo "  Input file: $INPUT_SCRIPT"
echo "  Log dir:   $LOG_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "=========================================="

> commands.txt
# The '-in' flag specifies the input script.
# The '-log' flag specifies the output log file.
# The '-var' flag allows passing a variable into the LAMMPS input script
for (( m=$MOLECULES; m<=${MOLECULES_END:-$MOLECULES}; m+=${MOLECULES_STEP:-1} )); do
  e="${VAR_EPSILON}"
  end="${VAR_EPSILON_END:-$VAR_EPSILON}"
  step="${VAR_EPSILON_STEP:-1}"
  # guard against zero step
  if [ "$(echo "$step == 0" | bc -l)" -eq 1 ]; then
    step="1"
  fi
  while [ "$(echo "$e <= $end" | bc -l)" -eq 1 ]; do
    EPSILON_VAL=$(printf "%.1f" "$e")
    FILENAME="${INPUT_SCRIPT}_${m}_${EPSILON_VAL}"
    LOG_FILE="${LOG_DIR}/${FILENAME}.log"
    echo "\"$LAMMPS_EXECUTABLE\" -in \"$INPUT_SCRIPT\" -log \"$LOG_FILE\" -var filename \"$FILENAME\" -var molecules \"$m\" -var var_epsilon \"$EPSILON_VAL\"" >> commands.txt
    e=$(echo "$e + $step" | bc -l)
  done
done

# Run commands in commands.txt in parallel based on CPU availability
if [[ -x "$(command -v nproc)" ]]; then
  MAX_JOBS=$(nproc)
else
  MAX_JOBS=1
fi

echo "Running simulations in parallel with up to $MAX_JOBS jobs..."
running_jobs=0
while read -r cmd; do
  FILENAME=$(echo "$cmd" | grep -oP '(?<=-var filename ")[^"]*')
  ( bash -c "$cmd" ; mv "${FILENAME}.lammpstrj" "$OUTPUT_DIR/" ) &
  running_jobs=$((running_jobs + 1))
  if [ $running_jobs -ge $MAX_JOBS ]; then
    wait
    running_jobs=0
  fi
done < commands.txt
wait
echo "All parallel jobs finished."

# Move any trajectory files left in the current directory to the output directory
for TRAJ_FILE in ./*.lammpstrj; do
  if [ -f "$TRAJ_FILE" ]; then
    mv "$TRAJ_FILE" "$OUTPUT_DIR/"
    echo "Moved trajectory file '$TRAJ_FILE' to '$OUTPUT_DIR/'"
  fi
done

# --- Post-simulation ---
echo ""
echo "=========================================="
echo "LAMMPS simulation finished."
echo "Check the log directory '$LOG_DIR' for details."
echo "Run the trajectory file with ovito: ovito '$OUTPUT_DIR/${INPUT_SCRIPT}_${MOLECULES}_${VAR_EPSILON}.lammpstrj' for visualization"
echo "=========================================="
