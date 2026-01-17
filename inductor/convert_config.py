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
        Tuple of (formatted ROCmGemmConfig string, config_tuple for deduplication)
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

        # Apply pruning rules if enabled
        if prune:
            # Set all waves_per_eu to 0
            waves_per_eu = 0
            # Set num_stages=3 to 2
            if num_stages == 3:
                num_stages = 2

        # Create a tuple for deduplication (based on final output values)
        config_tuple = (block_size_m, block_size_n, block_size_k, num_stages,
                       num_warps, group_size_m, waves_per_eu, kpack)

        # Format the output
        output = f"            ROCmGemmConfig({block_size_m}, {block_size_n}, {block_size_k}, {num_stages}, {num_warps}, group_m={group_size_m}, waves_per_eu={waves_per_eu}, kpack={kpack}),"

        return output, config_tuple

    except Exception as e:
        return f"# ERROR: Could not parse line: {e}", None


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

        for i, line in enumerate(lines, 1):
            line = line.strip()
            # Skip empty lines
            if not line:
                continue

            # Skip comment lines (if any)
            if line.startswith('#'):
                converted_lines.append(line)
                continue

            converted, config_tuple = convert_dict_to_config(line, prune=prune)

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

        if output_file:
            with open(output_file, 'w') as f:
                f.write(output_text)
                f.write('\n')
            print(f"Successfully converted {len(converted_lines)} unique lines")
            if duplicates_count > 0:
                print(f"Removed {duplicates_count} duplicate configurations")
            if prune:
                print(f"Pruning applied: waves_per_eu=0, num_stages=3->2")
            print(f"Output written to: {output_file}")
        else:
            print(output_text)

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
