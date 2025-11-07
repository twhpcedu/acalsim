#!/bin/bash

################################################################################
# Top Profiling Script for ACAL Simulator
#
# DESCRIPTION:
#   Monitors and captures real-time CPU and memory usage statistics for a running
#   executable using the Unix 'top' command. The script samples system metrics at
#   high frequency (5 samples per second) and converts the output to CSV format
#   for easy analysis in spreadsheet applications or data processing tools.
#
# PURPOSE:
#   This profiling tool helps developers and researchers:
#   - Monitor resource utilization during simulation runs
#   - Identify performance bottlenecks and memory leaks
#   - Generate time-series data for CPU/memory consumption analysis
#   - Validate that simulations stay within system resource constraints
#   - Compare resource usage across different configurations or workloads
#
# WORKFLOW:
#   1. Validates that required environment variables are set
#   2. Runs 'top' in batch mode at 200ms intervals for specified duration
#   3. Filters output to include only lines matching the target executable
#   4. Removes leading whitespace from each line for consistent formatting
#   5. Converts space-delimited fields to comma-separated values (CSV)
#   6. Cleans up intermediate text file, leaving only the CSV output
#
# ENVIRONMENT VARIABLES (REQUIRED):
#   EXE - Name or pattern of the executable to monitor
#         Must match the process name as it appears in 'top' output
#         Example: EXE="acalsim" or EXE="my_simulator"
#
# ENVIRONMENT VARIABLES (OPTIONAL):
#   TIME     - Duration of profiling in seconds (default: 10)
#              Total samples collected = TIME * 5
#              Example: TIME=30 collects 150 samples over 30 seconds
#
#   FILENAME - Base name for output files without extension (default: top-profiling)
#              Script generates: ${FILENAME}.csv
#              Example: FILENAME="sim_profile_run1"
#
# TOOLS USED:
#   top - System monitoring utility that provides real-time process information
#         Flags: -b (batch mode), -d (delay interval), -n (number of iterations)
#   grep - Text search utility to filter lines matching the executable name
#   sed - Stream editor to remove leading whitespace from each line
#   tr - Character translation utility to convert spaces to commas
#
# OUTPUT FORMAT:
#   Generates a CSV file with columns from 'top' output (column names may vary by system):
#   - PID: Process ID
#   - USER: Process owner
#   - PR: Priority
#   - NI: Nice value
#   - VIRT: Virtual memory size
#   - RES: Resident memory size (physical RAM usage)
#   - SHR: Shared memory size
#   - S: Process status (R=running, S=sleeping, etc.)
#   - %CPU: CPU usage percentage
#   - %MEM: Memory usage percentage
#   - TIME+: Cumulative CPU time
#   - COMMAND: Executable name
#
# USAGE EXAMPLES:
#   Basic usage with default settings (10 seconds, 50 samples):
#     $ EXE="acalsim" ./top-profiling.sh
#
#   Profile for 60 seconds with custom output filename:
#     $ EXE="acalsim" TIME=60 FILENAME="long_run_profile" ./top-profiling.sh
#
#   Monitor a specific process by PID pattern:
#     $ EXE="12345" TIME=5 ./top-profiling.sh
#
#   Profile with all custom parameters:
#     $ EXE="my_simulator" TIME=120 FILENAME="results/sim_metrics" ./top-profiling.sh
#
# EXPECTED OUTPUT:
#   Console output on success:
#     (silent - script completes without output)
#
#   Console output on error:
#     The environment variable EXE does not exits.
#
#   Generated files:
#     ${FILENAME}.csv - Comma-separated values with profiling data
#     (${FILENAME}.txt is created temporarily and then removed)
#
# CSV CONTENT EXAMPLE:
#   12345,user,20,0,4236800,128456,12345,S,45.2,1.5,10:23.45,acalsim
#   12345,user,20,0,4238900,129234,12345,R,48.1,1.5,10:24.89,acalsim
#   12345,user,20,0,4239100,129456,12345,S,42.3,1.5,10:26.12,acalsim
#
# SAMPLING RATE:
#   The script uses a 0.2 second (200ms) sampling interval, resulting in:
#   - 5 samples per second
#   - 300 samples per minute
#   - Default 10-second run produces 50 samples
#
#   This high-frequency sampling captures transient behavior and provides
#   detailed time-series data for statistical analysis.
#
# ANALYSIS WORKFLOW:
#   After generating the CSV file, typical analysis steps include:
#   1. Import CSV into spreadsheet software (Excel, LibreOffice, etc.)
#   2. Plot %CPU and %MEM over time to visualize resource usage trends
#   3. Calculate statistics: mean, max, standard deviation of CPU/memory
#   4. Identify anomalies: sudden spikes, memory leaks, CPU saturation
#   5. Compare profiles across different simulation configurations
#
# NOTES:
#   - The script requires the target process to be running when executed
#   - If the process name appears in multiple processes, all matches are captured
#   - Output format depends on 'top' version (may vary across Linux distributions)
#   - Very short TIME values (< 1 second) may produce incomplete data
#   - Large TIME values generate correspondingly large CSV files
#
# EXIT CODES:
#   0 - Success (profiling completed and CSV generated)
#   1 - Failure (EXE environment variable not set)
#
# AUTHOR:
#   Playlab/ACAL
#
# LICENSE:
#   Apache License, Version 2.0
################################################################################

# Set profiling duration in seconds (default: 10 seconds)
# Override with: TIME=30 ./top-profiling.sh
TIME="${TIME:-10}"

# Set output filename base (without extension)
# The script will create ${FILENAME}.csv
# Override with: FILENAME="my_profile" ./top-profiling.sh
FILENAME="${FILENAME:-top-profiling}"

# Validate that EXE environment variable is set
# This is the process name/pattern to monitor with 'top'
# Exit with error message if not provided
[[ -z "${EXE}" ]] && echo "The environment variable EXE does not exits." && exit 1

# Run 'top' in batch mode and capture output for the target executable
# -b: Batch mode (non-interactive, suitable for output redirection)
# -d 0.2: Sample every 0.2 seconds (5 samples per second)
# -n $((TIME * 5)): Total number of iterations (TIME seconds * 5 samples/second)
# | grep "${EXE}": Filter only lines containing the target executable name
# > "${FILENAME}.txt": Redirect filtered output to temporary text file
top -b -d 0.2 -n $(("${TIME}" * 5)) | grep "${EXE}" >"${FILENAME}.txt"

# Convert space-delimited output to CSV format
# sed 's/^[ \t]*//' : Remove leading spaces and tabs from each line
# | tr -s ' ' ',' : Translate sequences of spaces to single commas
# > "${FILENAME}.csv" : Write CSV-formatted output to final file
sed 's/^[ \t]*//' "${FILENAME}.txt" | tr -s ' ' ',' >"${FILENAME}.csv"

# Clean up intermediate text file (only CSV file is needed)
rm "${FILENAME}.txt"
