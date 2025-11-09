#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Category Analysis Script for ResearcherBench
Calculates average coverage, groundedness, and faithfulness scores per category
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def load_json(file_path: str) -> any:
    """Load JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def get_category_mapping(questions_file: str) -> Dict[int, str]:
    """
    Load questions and create mapping from question ID to category

    Returns:
        Dict mapping question_id -> category
    """
    questions = load_json(questions_file)
    if not questions:
        return {}

    category_map = {}
    for q in questions:
        category_map[q['id']] = q['category']

    return category_map


def load_coverage_scores(model_name: str, results_dir: str) -> Dict[int, float]:
    """
    Load coverage scores from rubric evaluation results

    Returns:
        Dict mapping question_id -> coverage_score (recall)
    """
    rubric_file = os.path.join(results_dir, 'rubric_eval', model_name,
                               f'{model_name}_evaluation_results.json')

    if not os.path.exists(rubric_file):
        print(f"Warning: Rubric evaluation file not found: {rubric_file}")
        return {}

    results = load_json(rubric_file)
    if not results:
        return {}

    coverage_scores = {}
    for result in results:
        question_id = result['id']
        coverage = result['result']['recall']
        coverage_scores[question_id] = coverage

    return coverage_scores


def load_factual_scores(model_name: str, results_dir: str) -> Dict[int, Tuple[float, float]]:
    """
    Load faithfulness and groundedness scores from factual evaluation results

    Returns:
        Dict mapping question_id -> (faithfulness_score, groundedness_score)
    """
    factual_file = os.path.join(results_dir, 'factual_eval', model_name, 'factual_results.json')

    if not os.path.exists(factual_file):
        print(f"Warning: Factual evaluation file not found: {factual_file}")
        return {}

    results = load_json(factual_file)
    if not results:
        return {}

    factual_scores = {}
    for result in results:
        question_id = result['id']
        faithfulness = result.get('faithfulness_score', 0.0)
        groundedness = result.get('groundedness_score', 0.0)
        factual_scores[question_id] = (faithfulness, groundedness)

    return factual_scores


def calculate_category_averages(category_map: Dict[int, str],
                                coverage_scores: Dict[int, float],
                                factual_scores: Dict[int, Tuple[float, float]]) -> Dict:
    """
    Calculate average scores per category

    Returns:
        Dict with category -> {coverage, faithfulness, groundedness, count}
    """
    category_data = defaultdict(lambda: {
        'coverage': [],
        'faithfulness': [],
        'groundedness': [],
    })

    # Collect all question IDs that have scores
    all_ids = set(coverage_scores.keys()) | set(factual_scores.keys())

    for question_id in all_ids:
        if question_id not in category_map:
            continue

        category = category_map[question_id]

        # Add coverage score if available
        if question_id in coverage_scores:
            category_data[category]['coverage'].append(coverage_scores[question_id])

        # Add factual scores if available
        if question_id in factual_scores:
            faithfulness, groundedness = factual_scores[question_id]
            category_data[category]['faithfulness'].append(faithfulness)
            category_data[category]['groundedness'].append(groundedness)

    # Calculate averages
    results = {}
    for category, data in category_data.items():
        results[category] = {
            'coverage': {
                'mean': sum(data['coverage']) / len(data['coverage']) if data['coverage'] else 0.0,
                'count': len(data['coverage'])
            },
            'faithfulness': {
                'mean': sum(data['faithfulness']) / len(data['faithfulness']) if data['faithfulness'] else 0.0,
                'count': len(data['faithfulness'])
            },
            'groundedness': {
                'mean': sum(data['groundedness']) / len(data['groundedness']) if data['groundedness'] else 0.0,
                'count': len(data['groundedness'])
            }
        }

    return results


def plot_results(model_name: str, category_results: Dict, output_file: str = None):
    """Create a bar plot visualization of category results"""

    # Category order (as in paper)
    category_order = ['Open Consulting', 'Literature Review', 'Technical Details']

    # Filter to only include categories that exist in results
    categories = [cat for cat in category_order if cat in category_results]

    if not categories:
        print("No data available to plot")
        return

    # Extract data for plotting
    coverage_data = [category_results[cat]['coverage']['mean'] for cat in categories]
    groundedness_data = [category_results[cat]['groundedness']['mean'] for cat in categories]
    faithfulness_data = [category_results[cat]['faithfulness']['mean'] for cat in categories]

    # Set up the plot
    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create bars
    bars1 = ax.bar(x - width, coverage_data, width, label='Coverage', alpha=0.8)
    bars2 = ax.bar(x, groundedness_data, width, label='Groundedness', alpha=0.8)
    bars3 = ax.bar(x + width, faithfulness_data, width, label='Faithfulness', alpha=0.8)

    # Customize plot
    ax.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Category Analysis for Model: {model_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)

    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)

    plt.tight_layout()

    # Save or show
    if output_file:
        # Generate plot filename from output file
        if output_file.endswith('.txt'):
            plot_file = output_file.replace('.txt', '.png')
        else:
            plot_file = output_file + '_plot.png'

        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {plot_file}")
    else:
        # Save to default location
        plot_file = f"results/{model_name}_category_plot.png"
        os.makedirs('results', exist_ok=True)
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {plot_file}")

    plt.close()


def print_results(model_name: str, category_results: Dict, output_file: str = None):
    """Print results in a formatted table"""

    output_lines = []

    # Header
    header = f"\n{'='*80}\n"
    header += f"Category Analysis for Model: {model_name}\n"
    header += f"{'='*80}\n"
    output_lines.append(header)

    # Category order (as in paper)
    category_order = ['Open Consulting', 'Literature Review', 'Technical Details']

    # Table header
    table_header = f"\n{'Category':<25} {'Coverage':>12} {'Groundedness':>15} {'Faithfulness':>15} {'N':>8}\n"
    table_header += f"{'-'*80}\n"
    output_lines.append(table_header)

    # Overall stats
    overall_coverage = []
    overall_faithfulness = []
    overall_groundedness = []

    # Print each category
    for category in category_order:
        if category not in category_results:
            continue

        data = category_results[category]

        coverage_mean = data['coverage']['mean']
        faithfulness_mean = data['faithfulness']['mean']
        groundedness_mean = data['groundedness']['mean']

        # Use coverage count as primary (should be same for all)
        count = data['coverage']['count']

        row = f"{category:<25} {coverage_mean:>12.4f} {groundedness_mean:>15.4f} {faithfulness_mean:>15.4f} {count:>8}\n"
        output_lines.append(row)

        # Collect for overall average
        overall_coverage.extend([coverage_mean] * count)
        overall_faithfulness.extend([faithfulness_mean] * count)
        overall_groundedness.extend([groundedness_mean] * count)

    # Overall average
    if overall_coverage:
        total_count = sum(data['coverage']['count'] for data in category_results.values())

        # Weighted average across categories
        overall_cov = sum(data['coverage']['mean'] * data['coverage']['count']
                         for data in category_results.values()) / total_count
        overall_ground = sum(data['groundedness']['mean'] * data['groundedness']['count']
                            for data in category_results.values()) / total_count
        overall_faith = sum(data['faithfulness']['mean'] * data['faithfulness']['count']
                           for data in category_results.values()) / total_count

        separator = f"{'-'*80}\n"
        overall_row = f"{'Overall':<25} {overall_cov:>12.4f} {overall_ground:>15.4f} {overall_faith:>15.4f} {total_count:>8}\n"
        output_lines.append(separator)
        output_lines.append(overall_row)

    footer = f"{'='*80}\n"
    output_lines.append(footer)

    # Print to console
    for line in output_lines:
        print(line, end='')

    # Save to file if specified
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.writelines(output_lines)
            print(f"\nResults saved to: {output_file}")
        except Exception as e:
            print(f"Error saving results to file: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Analyze evaluation results by category',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python category_analysis.py --model wiona1
  python category_analysis.py --model wiona2 --output results/wiona2_category_analysis.txt
  python category_analysis.py --model wiona1 --plot
  python category_analysis.py --model OpenAI --data_dir ./data --results_dir ./results --plot
        """
    )

    parser.add_argument('--model', type=str, required=True,
                        help='Model name (must match directory name in results/)')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data directory containing eval_data/questions.json (default: data)')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Results directory (default: results)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (optional, prints to console if not specified)')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Generate a bar plot visualization of the results (default: False)')

    args = parser.parse_args()

    # File paths
    questions_file = os.path.join(args.data_dir, 'eval_data', 'questions.json')

    # Check if files exist
    if not os.path.exists(questions_file):
        print(f"Error: Questions file not found: {questions_file}")
        return 1

    # Load category mapping
    print(f"Loading category mapping from {questions_file}...")
    category_map = get_category_mapping(questions_file)
    if not category_map:
        print("Error: Failed to load category mapping")
        return 1

    print(f"Loaded {len(category_map)} questions with categories")

    # Load coverage scores
    print(f"\nLoading coverage scores for model: {args.model}...")
    coverage_scores = load_coverage_scores(args.model, args.results_dir)
    print(f"Loaded coverage scores for {len(coverage_scores)} questions")

    # Load factual scores
    print(f"Loading factual scores for model: {args.model}...")
    factual_scores = load_factual_scores(args.model, args.results_dir)
    print(f"Loaded factual scores for {len(factual_scores)} questions")

    # Calculate category averages
    print(f"\nCalculating category averages...")
    category_results = calculate_category_averages(category_map, coverage_scores, factual_scores)

    # Print results
    print_results(args.model, category_results, args.output)

    # Generate plot if requested
    if args.plot:
        print(f"\nGenerating plot...")
        plot_results(args.model, category_results, args.output)

    return 0


if __name__ == '__main__':
    sys.exit(main())
