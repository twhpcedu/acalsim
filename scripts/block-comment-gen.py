#!/usr/bin/env python3
"""C++ Block Comment Generator.

This script generates aesthetically formatted C++ block comments with centered titles.
It creates comment blocks with configurable width and padding, perfect for section
headers, separators, and important code annotations in C++ source files.

The generator uses a unit-based width system to ensure consistent alignment across
different terminals and editors. Comments are centered within the block and surrounded
by padding for visual clarity.

Algorithm:
    1. Calculate the required inner width based on title length and padding
    2. Round up to the nearest multiple of unit_width for consistent sizing
    3. Center the title text within the calculated width
    4. Generate a 5-line block comment structure with top/bottom borders and padding

Usage Examples:
    Basic usage with default settings (32-char units, 4-char min padding):
        $ python block-comment-gen.py --title "My Section"

    Custom width units for wider blocks:
        $ python block-comment-gen.py --title "Database Layer" --unit-width 40

    Tighter padding for compact comments:
        $ python block-comment-gen.py --title "TODO" --unit-width 20 --min-pad 2

    Long title requiring multiple units:
        $ python block-comment-gen.py --title "This is a very long section title"

Example Outputs:
    For --title "Hello World" (default settings):
        /********************************
         *                              *
         *         Hello World          *
         *                              *
         ********************************/

    For --title "Init" --unit-width 16 --min-pad 2:
        /******************
         *                *
         *      Init      *
         *                *
         ******************/

    For --title "A Very Long Title" --unit-width 32:
        /****************************************************************
         *                                                              *
         *                      A Very Long Title                       *
         *                                                              *
         ****************************************************************/

Notes:
    - The script outputs directly to stdout, making it pipeable
    - Inner width is always a multiple of unit_width for consistency
    - Title is automatically centered with equal spacing on both sides
    - Works with any ASCII text; Unicode characters may affect alignment

Author: Playlab/ACAL
License: Apache License 2.0
"""

# Copyright 2023-2025 Playlab/ACAL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import click


@click.command()
@click.option('--title', help='The content of the comment block.', required=True, type=str)
@click.option('--unit-width', help='The basic unit of the block width.', default=32, type=int)
@click.option(
    '--min-pad',
    help='The minimum margin width on the left and right of the title respectively.',
    default=4,
    type=int
)
def gen_cpp_comment(title: str, unit_width: int, min_pad: int) -> None:
	"""Generate a centered C++ block comment with the specified title.

	This function creates a formatted C++ block comment with the title centered
	within a box of asterisks. The box width is calculated to be the smallest
	multiple of unit_width that can accommodate the title with minimum padding
	on both sides.

	The generated comment block has the following structure:
	    Line 1: Top border (/* followed by asterisks)
	    Line 2: Empty padding line with side borders
	    Line 3: Title line with centered text
	    Line 4: Empty padding line with side borders
	    Line 5: Bottom border (asterisks followed by */)

	Width Calculation:
	    The inner width is computed as:
	        inner_width = unit_width * ceil((title_length + 2 * min_pad) / unit_width)

	    This ensures:
	    - The title fits with at least min_pad spaces on each side
	    - The total width is always a multiple of unit_width
	    - Consistent sizing across different title lengths

	Centering Algorithm:
	    The title is centered by calculating the starting position:
	        start_index = (inner_width - title_length) / 2
	    Spaces are then distributed:
	    - Left padding: start_index spaces
	    - Title text: actual title string
	    - Right padding: remaining spaces to fill inner_width

	Args:
	    title (str): The text to display in the center of the comment block.
	        Can be any string, though ASCII characters are recommended for
	        proper alignment. Length affects the final block width.

	    unit_width (int): The basic unit for width calculation. The final
	        inner width will be a multiple of this value. Defaults to 32.
	        Common values:
	        - 16: Compact blocks for short comments
	        - 32: Standard blocks (default)
	        - 40-64: Wide blocks for longer titles or visual prominence

	    min_pad (int): Minimum number of spaces required between the title
	        and the left/right borders. Defaults to 4. Larger values create
	        more whitespace around the title. Must be >= 0.

	Returns:
	    None: The function prints the formatted comment block directly to stdout.

	Prints:
	    A 5-line C++ block comment formatted as described above. Each line
	    ends with a newline character.

	Examples:
	    >>> gen_cpp_comment("Init", 32, 4)
	    /********************************
	     *                              *
	     *            Init              *
	     *                              *
	     ********************************/

	    >>> gen_cpp_comment("Short", 16, 2)
	    /******************
	     *                *
	     *     Short      *
	     *                *
	     ******************/

	    >>> gen_cpp_comment("Configuration Manager", 32, 4)
	    /****************************************************************
	     *                                                              *
	     *                   Configuration Manager                      *
	     *                                                              *
	     ****************************************************************/

	Notes:
	    - The function uses integer division for centering, which may result
	      in an extra space on the right for odd-length titles
	    - Output is sent to stdout, making it easy to redirect or pipe
	    - No validation is performed on input parameters; invalid values
	      may produce malformed output
	    - The function is decorated with Click options for CLI usage

	Raises:
	    No exceptions are raised directly by this function, though invalid
	    numeric inputs will be caught by Click's type validation.
	"""
	# Calculate the total inner width needed to fit the title with padding.
	# The width is rounded up to the nearest multiple of unit_width to maintain
	# consistent sizing across different title lengths.
	inner_width: int = unit_width * math.ceil((len(title) + min_pad * 2) / unit_width)

	# Calculate the starting index for the title to achieve centering.
	# This positions the title so there's equal spacing on both sides (or nearly equal
	# for odd-length titles, where the right side gets one extra space).
	title_start_idx: int = int((inner_width / 2) - (len(title) / 2))

	# Print the top line: opening comment delimiter followed by a full line of asterisks
	print('/*' + '*' * inner_width + '*')

	# Print the 1st padding line: creates vertical spacing above the title
	print(' *' + ' ' * inner_width + '*')

	# Print the title line: centered text with calculated left padding and
	# right padding to fill the remaining space
	print(
	    ' *' + ' ' * title_start_idx + title + ' ' * (inner_width - len(title) - title_start_idx) +
	    '*'
	)

	# Print the 2nd padding line: creates vertical spacing below the title
	print(' *' + ' ' * inner_width + '*')

	# Print the bottom line: full line of asterisks followed by closing comment delimiter
	print(' *' + '*' * inner_width + '*/')


if __name__ == '__main__':
	gen_cpp_comment()
