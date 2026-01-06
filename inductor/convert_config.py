#!/usr/bin/env python3
"""
Convert dictionary-formatted strings to ROCmGemmConfig format.

Input format: Dictionary strings (one per line)
Output format: ROCmGemmConfig constructor calls
"""

import sys
import ast


def convert_dict_to_config(dict_str):
    """
    Convert a dictionary string to ROCmGemmConfig format.
    
    Args:
        dict_str: String representation of a dictionary
        
    Returns:
        Formatted ROCmGemmConfig string
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
        
        # Format the output
        output = f"            ROCmGemmConfig({block_size_m}, {block_size_n}, {block_size_k}, {num_stages}, {num_warps}, group_m={group_size_m}, waves_per_eu={waves_per_eu}, kpack={kpack}),"
        
        return output
        
    except Exception as e:
        return f"# ERROR: Could not parse line: {e}"


def main():
    """
    Main function to read input file and convert each line.
    """
    if len(sys.argv) < 2:
        print("Usage: python convert_config.py <input_file> [output_file]")
        print("\nIf output_file is not provided, results will be printed to stdout")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        # Read input file
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        # Convert each line
        converted_lines = []
        for i, line in enumerate(lines, 1):
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            
            # Skip comment lines (if any)
            if line.startswith('#'):
                converted_lines.append(line)
                continue
            
            converted = convert_dict_to_config(line)
            converted_lines.append(converted)
        
        # Output results
        output_text = '\n'.join(converted_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(output_text)
                f.write('\n')
            print(f"Successfully converted {len(converted_lines)} lines")
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
