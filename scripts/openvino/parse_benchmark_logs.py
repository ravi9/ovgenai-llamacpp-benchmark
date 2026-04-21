#!/usr/bin/env python3
"""
Parse llama-bench and llm_bench CSV files from logs folder and create a consolidated CSV.
"""

import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Optional
import statistics


def extract_quantization_from_filename(filename: str) -> str:
    """Extract quantization type from model filename (e.g., Q4_0, Q4_K_M, etc.)"""
    # Look for patterns like Q4_0, Q4_K_M, Q8_0, etc.
    match = re.search(r'Q\d+_[0-9A-Z_]+', filename)
    if match:
        return match.group(0)
    # Check if it's an FP16 or BF16 model
    if 'fp16' in filename.lower() or 'f16' in filename.lower():
        return 'FP16'
    if 'bf16' in filename.lower():
        return 'BF16'
    return 'UNKNOWN'


def get_framework_tag(filename: str, device: str) -> str:
    """Map filename to framework tag"""
    filename_lower = filename.lower()
    
    if 'llama_ggml' in filename_lower:
        return f'llama.cpp/ggml/{device}'
    elif 'llama_vulkan' in filename_lower:
        return f'llama.cpp/vulkan/{device}'
    elif 'llama_ov' in filename_lower:
        return f'llama.cpp/openvino/{device}'
    elif 'ov_genai_gguf' in filename_lower:
        return f'openvino.genai/gguf/{device}'
    elif 'ov_genai_ir' in filename_lower:
        # Extract optimization mode from filename
        if '_cw_' in filename_lower:
            opt = 'CW'
        elif '_gs32_' in filename_lower:
            opt = 'GS32'
        elif '_default_' in filename_lower:
            opt = 'DEFAULT'
        else:
            opt = 'UNKNOWN'
        return f'openvino.genai/ir/{opt}/{device}'
    
    return 'UNKNOWN'


def parse_llama_bench_csv(filepath: Path, model_name: str) -> List[Dict]:
    """Parse llama-bench CSV file and extract metrics"""
    results = []
    
    # Read file and skip any non-CSV lines at the beginning
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Check if there's an error message
    has_error = any('error' in line.lower() for line in lines[:5])
    
    # Find the CSV header line (starts with "build_commit")
    csv_start = 0
    for i, line in enumerate(lines):
        if line.startswith('build_commit'):
            csv_start = i
            break
    
    # Parse CSV from the header line onwards
    csv_lines = lines[csv_start:]
    if not csv_lines:
        return results
    
    reader = csv.DictReader(csv_lines)
    try:
        rows = list(reader)
    except Exception as e:
        return results
    
    # Determine device from filename
    filename = filepath.name.lower()
    if 'cpu' in filename:
        device = 'CPU'
    elif 'gpu' in filename:
        device = 'GPU'
    elif 'npu' in filename:
        device = 'NPU'
    else:
        device = 'UNKNOWN'
    
    framework = get_framework_tag(filepath.name, device)
    
    # Handle case where benchmark failed (has error, no data rows)
    if not rows or (has_error and all(not row.get('model_filename') for row in rows)):
        # Try to extract model info from the error message or path
        model_filename = None
        for line in lines:
            if 'models/gguf/' in line:
                match = re.search(r'models/gguf/([^\s\'\"]+\.gguf)', line)
                if match:
                    model_filename = match.group(1)
                    break
        
        if not model_filename:
            # Use model_name from folder
            model_filename = f"{model_name}.gguf"
        
        quantization = extract_quantization_from_filename(model_filename)
        
        result = {
            'Model': model_name,
            'Framework': framework,
            'Device': device,
            'Quantization': quantization,
            'NUM_INPUT_TOKENS': 'FAILED',
            'NUM_OUTPUT_TOKENS': 'FAILED',
            'LOAD_COMPILE_TIME (ms)': 'FAILED',
            'TTFT (ms)': 'FAILED',
            'PROMPT_EVAL / PP (tok/s)': 'FAILED',
            'TOKEN_GEN (tok/s)': 'FAILED',
            'AVG_TOKEN_LATENCY (ms/tok)': 'FAILED',
            'GENERATION_TIME (s)': 'FAILED',
            'MEMORY_RSS (GiB)': 'FAILED',
            'MEMORY_SYS (GiB)': 'FAILED',
            'COMPILE_MEMORY (GiB)': 'FAILED',
            'TG_STDDEV (tok/s)': 'FAILED',
            'llama.cpp commit': '',
            'OV/GenAI version': '',
        }
        results.append(result)
        return results
    
    # Extract common fields from first row
    first_row = rows[0]
    model_filename = os.path.basename(first_row['model_filename'])
    quantization = extract_quantization_from_filename(model_filename)
    build_commit = first_row.get('build_commit', 'UNKNOWN')
    
    # Find rows for pp (prompt processing), tg32, tg128
    pp_row = None
    tg_row = None
    
    for row in rows:
        n_prompt = int(row['n_prompt'])
        n_gen = int(row['n_gen'])
        n_depth = int(row['n_depth'])
        
        # PP row: n_prompt > 0, n_gen = 0, n_depth = 0
        if n_prompt > 0 and n_gen == 0 and n_depth == 0:
            pp_row = row
        # TG row: n_prompt = 0, n_gen > 0, n_depth = 0
        elif n_prompt == 0 and n_gen > 0 and n_depth == 0:
            tg_row = row
    
    if pp_row and tg_row:
        n_prompt = int(pp_row['n_prompt'])
        n_gen = int(tg_row['n_gen'])
        
        # Calculate metrics
        avg_ts_pp = float(pp_row['avg_ts'])
        avg_ts_tg = float(tg_row['avg_ts'])
        stddev_ts_tg = float(tg_row['stddev_ts'])
        avg_ns_tg = float(tg_row['avg_ns'])
        
        # TTFT = (n_prompt / avg_ts) * 1000
        ttft_ms = (n_prompt / avg_ts_pp) * 1000 if avg_ts_pp > 0 else 0
        
        # Prompt eval throughput
        prompt_eval_toks = avg_ts_pp
        
        # Token generation throughput
        token_gen_toks = avg_ts_tg
        
        # Average token latency
        avg_token_latency = 1000 / avg_ts_tg if avg_ts_tg > 0 else 0
        
        # Generation time
        generation_time_s = avg_ns_tg / 1e9
        
        result = {
            'Model': model_name,
            'Framework': framework,
            'Device': device,
            'Quantization': quantization,
            'NUM_INPUT_TOKENS': n_prompt,
            'NUM_OUTPUT_TOKENS': n_gen,
            'LOAD_COMPILE_TIME (ms)': '',  # Not available in llama-bench CSV
            'TTFT (ms)': f'{ttft_ms:.2f}',
            'PROMPT_EVAL / PP (tok/s)': f'{prompt_eval_toks:.2f}',
            'TOKEN_GEN (tok/s)': f'{token_gen_toks:.2f}',
            'AVG_TOKEN_LATENCY (ms/tok)': f'{avg_token_latency:.2f}',
            'GENERATION_TIME (s)': f'{generation_time_s:.2f}',
            'MEMORY_RSS (GiB)': '',  # Not available
            'MEMORY_SYS (GiB)': '',  # Not available
            'COMPILE_MEMORY (GiB)': '',  # Not available
            'TG_STDDEV (tok/s)': f'{stddev_ts_tg:.2f}',
            'llama.cpp commit': build_commit,
            'OV/GenAI version': '',
        }
        
        results.append(result)
    
    return results


def parse_llm_bench_csv(filepath: Path, model_name: str) -> List[Dict]:
    """Parse llm_bench (OpenVINO GenAI) CSV file and extract metrics"""
    results = []
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        return results
    
    # Filter out non-numeric iteration rows and empty rows
    data_rows = []
    avg_row = None
    mini_row = None
    median_row = None
    
    for row in rows:
        iteration = row.get('iteration', '').strip()
        if iteration == '':
            continue
        elif iteration == 'avg':
            avg_row = row
        elif iteration == 'mini':
            mini_row = row
        elif iteration == 'median':
            median_row = row
        elif iteration.isdigit():
            data_rows.append(row)
    
    if not avg_row:
        return results
    
    # Extract common fields
    model = avg_row.get('model', model_name)
    framework_raw = avg_row.get('framework', '')
    device = avg_row.get('device', 'UNKNOWN')
    precision = avg_row.get('precision', '')
    
    # Determine device from filename if not in CSV
    filename = filepath.name.lower()
    if not device or device == 'UNKNOWN':
        if 'cpu' in filename:
            device = 'CPU'
        elif 'gpu' in filename:
            device = 'GPU'
        elif 'npu' in filename:
            device = 'NPU'
    
    framework = get_framework_tag(filepath.name, device)
    
    # Extract quantization from precision or model name
    quantization = precision if precision else extract_quantization_from_filename(model)
    
    # Get metrics from avg row
    try:
        input_size = float(avg_row.get('input_size', 0))
        output_size = float(avg_row.get('output_size', 0))
        pretrain_time_s = float(avg_row.get('pretrain_time(s)', 0)) if avg_row.get('pretrain_time(s)') else 0
        first_latency_ms = float(avg_row.get('1st_latency(ms)', 0)) if avg_row.get('1st_latency(ms)') else 0
        second_avg_latency_ms = float(avg_row.get('2nd_avg_latency(ms)', 0)) if avg_row.get('2nd_avg_latency(ms)') else 0
        generation_time_s = float(avg_row.get('generation_time(s)', 0)) if avg_row.get('generation_time(s)') else 0
        
        # Memory metrics (convert MiB to GiB)
        max_rss_mem_mib = avg_row.get('max_rss_mem(MiB)', '')
        max_sys_mem_mib = avg_row.get('max_sys_mem(MiB)', '')
        compile_max_rss_mem_mib = avg_row.get('compile_max_rss_mem(MiB)', '')
        
        max_rss_mem_gib = float(max_rss_mem_mib) / 1024 if max_rss_mem_mib else 0
        max_sys_mem_gib = float(max_sys_mem_mib) / 1024 if max_sys_mem_mib else 0
        compile_mem_gib = float(compile_max_rss_mem_mib) / 1024 if compile_max_rss_mem_mib else 0
        
        # Calculate derived metrics
        load_compile_time_ms = pretrain_time_s * 1000
        ttft_ms = first_latency_ms
        prompt_eval_toks = (input_size / (first_latency_ms / 1000)) if first_latency_ms > 0 else 0
        token_gen_toks = (1000 / second_avg_latency_ms) if second_avg_latency_ms > 0 else 0
        avg_token_latency = second_avg_latency_ms
        
        # Calculate TG stddev from avg/mini/median rows
        tg_stddev = 0
        if mini_row and median_row:
            try:
                mini_2nd_latency = float(mini_row.get('2nd_avg_latency(ms)', 0)) if mini_row.get('2nd_avg_latency(ms)') else 0
                median_2nd_latency = float(median_row.get('2nd_avg_latency(ms)', 0)) if median_row.get('2nd_avg_latency(ms)') else 0
                
                # Compute tok/s for each iteration
                tg_values = []
                for row in data_rows:
                    if row.get('iteration', '').isdigit() and int(row['iteration']) > 0:  # Exclude warm-up
                        lat = float(row.get('2nd_avg_latency(ms)', 0)) if row.get('2nd_avg_latency(ms)') else 0
                        if lat > 0:
                            tg_values.append(1000 / lat)
                
                if len(tg_values) > 1:
                    tg_stddev = statistics.stdev(tg_values)
            except:
                pass
        
        result = {
            'Model': model_name,
            'Framework': framework,
            'Device': device,
            'Quantization': quantization,
            'NUM_INPUT_TOKENS': int(input_size),
            'NUM_OUTPUT_TOKENS': int(output_size),
            'LOAD_COMPILE_TIME (ms)': f'{load_compile_time_ms:.2f}',
            'TTFT (ms)': f'{ttft_ms:.2f}',
            'PROMPT_EVAL / PP (tok/s)': f'{prompt_eval_toks:.2f}',
            'TOKEN_GEN (tok/s)': f'{token_gen_toks:.2f}',
            'AVG_TOKEN_LATENCY (ms/tok)': f'{avg_token_latency:.2f}',
            'GENERATION_TIME (s)': f'{generation_time_s:.2f}',
            'MEMORY_RSS (GiB)': f'{max_rss_mem_gib:.2f}' if max_rss_mem_gib > 0 else '',
            'MEMORY_SYS (GiB)': f'{max_sys_mem_gib:.2f}' if max_sys_mem_gib > 0 else '',
            'COMPILE_MEMORY (GiB)': f'{compile_mem_gib:.2f}' if compile_mem_gib > 0 else '',
            'TG_STDDEV (tok/s)': f'{tg_stddev:.2f}' if tg_stddev > 0 else '',
            'llama.cpp commit': '',
            'OV/GenAI version': '',  # Would need to be extracted from log files or environment
        }
        
        results.append(result)
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
    
    return results


def read_commit_info(logs_dir: Path) -> tuple:
    """Read commit information from text files"""
    llama_commit = ''
    ov_commit = ''
    
    llama_commit_file = logs_dir / 'llama_cpp_commit.txt'
    if llama_commit_file.exists():
        with open(llama_commit_file, 'r') as f:
            llama_commit = f.read().strip().split()[0]
    
    ov_commit_file = logs_dir / 'openvino_genai_commit.txt'
    if ov_commit_file.exists():
        with open(ov_commit_file, 'r') as f:
            ov_commit = f.read().strip().split()[0]
    
    return llama_commit, ov_commit


def main():
    # Get the logs directory
    script_dir = Path(__file__).parent
    logs_dir = script_dir / 'logs'
    
    if not logs_dir.exists():
        print(f"Error: logs directory not found at {logs_dir}")
        return
    
    # Read commit info
    llama_commit, ov_commit = read_commit_info(logs_dir)
    
    # Collect all results
    all_results = []
    
    # Iterate through subdirectories in logs
    for subdir in sorted(logs_dir.iterdir()):
        if not subdir.is_dir():
            continue
        
        model_name = subdir.name
        print(f"Processing {model_name}...")
        
        # Process llama-bench files (*.log files that are CSV)
        for log_file in subdir.glob('llama_*.log'):
            try:
                results = parse_llama_bench_csv(log_file, model_name)
                # Add commit info (but not for failed benchmarks)
                for result in results:
                    if result['NUM_INPUT_TOKENS'] != 'FAILED':  # Don't add commit to failed benchmarks
                        if result['llama.cpp commit'] == 'UNKNOWN' or not result['llama.cpp commit']:
                            result['llama.cpp commit'] = llama_commit
                all_results.extend(results)
            except Exception as e:
                print(f"  Error processing {log_file.name}: {e}")
        
        # Process llm_bench CSV files
        for csv_file in subdir.glob('ov_genai_*.csv'):
            try:
                results = parse_llm_bench_csv(csv_file, model_name)
                # Add commit info
                for result in results:
                    result['OV/GenAI version'] = ov_commit
                all_results.extend(results)
            except Exception as e:
                print(f"  Error processing {csv_file.name}: {e}")
    
    # Write output CSV
    if all_results:
        output_file = script_dir / 'benchmark_results.csv'
        
        fieldnames = [
            'Model', 'Framework', 'Device', 'Quantization',
            'NUM_INPUT_TOKENS', 'NUM_OUTPUT_TOKENS',
            'LOAD_COMPILE_TIME (ms)', 'TTFT (ms)',
            'PROMPT_EVAL / PP (tok/s)', 'TOKEN_GEN (tok/s)',
            'AVG_TOKEN_LATENCY (ms/tok)', 'GENERATION_TIME (s)',
            'MEMORY_RSS (GiB)', 'MEMORY_SYS (GiB)', 'COMPILE_MEMORY (GiB)',
            'TG_STDDEV (tok/s)', 'llama.cpp commit', 'OV/GenAI version'
        ]
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\nSuccess! Wrote {len(all_results)} results to {output_file}")
    else:
        print("\nNo results found!")


if __name__ == '__main__':
    main()
