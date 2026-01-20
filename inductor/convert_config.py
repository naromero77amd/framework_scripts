#!/usr/bin/env python3
"""
Convert dictionary-formatted strings to ROCmGemmConfig format.

Input format: Dictionary strings (one per line)
Output format: ROCmGemmConfig constructor calls
"""

import sys
import ast
import argparse


def convert_dict_to_config(dict_str, prune=False):
    """
    Convert a dictionary string to ROCmGemmConfig format.

    Args:
        dict_str: String representation of a dictionary
        prune: If True, apply pruning rules to reduce configs

    Returns:
        Tuple of (formatted ROCmGemmConfig string, config_tuple for deduplication, was_pruned)
    """
    try:
        # Remove leading "- " if present (common in YAML-style lists)
        dict_str = dict_str.strip()
        if dict_str.startswith('- '):
            dict_str = dict_str[2:].strip()

        # Parse the string as a Python dictionary
        config_dict = ast.literal_eval(dict_str)

        # Extract required values
        block_size_m = config_dict.get('BLOCK_SIZE_M')
        block_size_n = config_dict.get('BLOCK_SIZE_N')
        block_size_k = config_dict.get('BLOCK_SIZE_K')
        num_stages = config_dict.get('num_stages')
        num_warps = config_dict.get('num_warps')
        group_size_m = config_dict.get('GROUP_SIZE_M')
        waves_per_eu = config_dict.get('waves_per_eu')
        kpack = config_dict.get('kpack')

        # Check if all required fields are present
        required_fields = {
            'BLOCK_SIZE_M': block_size_m,
            'BLOCK_SIZE_N': block_size_n,
            'BLOCK_SIZE_K': block_size_k,
            'num_stages': num_stages,
            'num_warps': num_warps,
            'GROUP_SIZE_M': group_size_m,
            'waves_per_eu': waves_per_eu,
            'kpack': kpack
        }

        missing = [k for k, v in required_fields.items() if v is None]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        # Track if this config was modified by pruning
        was_pruned = False

        # Apply pruning rules if enabled
        if prune:
            # Check if waves_per_eu will be changed
            if waves_per_eu != 0:
                was_pruned = True
            waves_per_eu = 0

            # Set num_stages=3 to 2
            if num_stages == 3:
                num_stages = 2
                was_pruned = True

        # Create a tuple for deduplication (based on final output values)
        config_tuple = (block_size_m, block_size_n, block_size_k, num_stages,
                       num_warps, group_size_m, waves_per_eu, kpack)

        # Format the output
        output = f"            ROCmGemmConfig({block_size_m}, {block_size_n}, {block_size_k}, {num_stages}, {num_warps}, group_m={group_size_m}, waves_per_eu={waves_per_eu}, kpack={kpack}),"

        return output, config_tuple, was_pruned

    except Exception as e:
        return f"# ERROR: Could not parse line: {e}", None, False


def main():
    """
    Main function to read input file and convert each line.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Convert dictionary-formatted strings to ROCmGemmConfig format.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('input_file', help='Input file containing dictionary configurations')
    parser.add_argument('output_file', nargs='?', default=None,
                        help='Output file (optional, prints to stdout if not provided)')
    parser.add_argument('--prune', action='store_true',
                        help='Apply pruning rules: set waves_per_eu=0 and num_stages=3 to 2')

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    prune = args.prune

    try:
        # Read input file
        with open(input_file, 'r') as f:
            lines = f.readlines()

        # Convert each line and track unique configs
        converted_lines = []
        seen_configs = set()  # Track unique configurations
        duplicates_count = 0
        total_processed = 0
        pruned_count = 0

        for i, line in enumerate(lines, 1):
            line = line.strip()
            # Skip empty lines
            if not line:
                continue

            # Skip comment lines (if any)
            if line.startswith('#'):
                converted_lines.append(line)
                continue

            total_processed += 1
            converted, config_tuple, was_pruned = convert_dict_to_config(line, prune=prune)

            # Track pruning
            if was_pruned:
                pruned_count += 1

            # Skip if this is an error
            if config_tuple is None:
                converted_lines.append(converted)
                continue

            # Check for duplicates
            if config_tuple in seen_configs:
                duplicates_count += 1
                continue  # Skip this duplicate

            # Add to seen configs and output
            seen_configs.add(config_tuple)
            converted_lines.append(converted)

        # Output results
        output_text = '\n'.join(converted_lines)

        # Print statistics to stderr (so they don't interfere with stdout output)
        print(f"Total configs processed: {total_processed}", file=sys.stderr)
        print(f"Successfully converted {len(converted_lines)} unique lines", file=sys.stderr)
        if duplicates_count > 0:
            print(f"Removed {duplicates_count} duplicate configurations", file=sys.stderr)
        if prune:
            print(f"Pruning applied: waves_per_eu=0, num_stages=3->2", file=sys.stderr)
            print(f"Configs modified by pruning: {pruned_count}", file=sys.stderr)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(output_text)
                f.write('\n')
            print(f"Output written to: {output_file}", file=sys.stderr)
        else:
            print(output_text)

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
