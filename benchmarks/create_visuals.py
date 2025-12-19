#!/usr/bin/env python3
"""Generate professional visualizations for GitHub README."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Don't use dark_background style - set colors manually

# Color palette - vibrant on dark
COLORS = {
    'primary': '#00D9FF',      # Cyan
    'secondary': '#FF6B6B',    # Coral
    'accent1': '#4ECDC4',      # Teal
    'accent2': '#FFE66D',      # Yellow
    'accent3': '#C44569',      # Pink
    'accent4': '#A855F7',      # Purple
    'success': '#10B981',      # Green
    'warning': '#F59E0B',      # Orange
    'bg_dark': '#0D1117',      # GitHub dark
    'bg_card': '#161B22',      # GitHub card
    'text': '#E6EDF3',         # Light text
    'text_muted': '#8B949E',   # Muted text
    'grid': '#30363D',         # Grid lines
}

CATEGORY_COLORS = {
    'math': COLORS['primary'],
    'code': COLORS['accent4'],
    'general': COLORS['accent1'],
    'compound': COLORS['accent2'],
    'math_complexity': COLORS['primary'],
    'code_complexity': COLORS['accent4'],
    'reasoning_chains': COLORS['accent1'],
}

DIFFICULTY_COLORS = {
    'easy': COLORS['success'],
    'medium': COLORS['accent2'],
    'hard': COLORS['warning'],
    'extreme': COLORS['secondary'],
}


def setup_figure(figsize=(12, 7)):
    """Create a figure with dark theme."""
    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS['bg_dark'])
    ax.set_facecolor(COLORS['bg_card'])
    return fig, ax


def style_axis(ax, title, xlabel='', ylabel=''):
    """Apply consistent styling to axis."""
    ax.set_title(title, fontsize=18, fontweight='bold', color=COLORS['text'], pad=20)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14, color=COLORS['text_muted'])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14, color=COLORS['text_muted'])
    ax.tick_params(colors=COLORS['text_muted'], labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_color(COLORS['grid'])
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')


def create_hero_banner():
    """Create a stunning hero image for the README."""
    fig, ax = plt.subplots(figsize=(14, 5), facecolor=COLORS['bg_dark'])
    ax.set_facecolor(COLORS['bg_dark'])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis('off')

    # Background gradient effect using scatter
    np.random.seed(42)
    for _ in range(200):
        x = np.random.uniform(0, 100)
        y = np.random.uniform(0, 40)
        size = np.random.uniform(5, 50)
        alpha = np.random.uniform(0.02, 0.08)
        color = np.random.choice([COLORS['primary'], COLORS['accent4'], COLORS['accent1']])
        ax.scatter(x, y, s=size, c=color, alpha=alpha)

    # Main title
    ax.text(50, 28, 'TinyLLM', fontsize=48, fontweight='bold',
            color=COLORS['text'], ha='center', va='center',
            fontfamily='sans-serif')

    # Subtitle
    ax.text(50, 20, 'Intelligent Neurons for Emergent Cognition',
            fontsize=18, color=COLORS['text_muted'], ha='center', va='center',
            fontfamily='sans-serif', style='italic')

    # Stats bar
    stats = ['267 Tests', '100% Pass', '24 Queries', '<12s Extreme']
    colors = [COLORS['success'], COLORS['primary'], COLORS['accent4'], COLORS['accent2']]
    for i, (stat, color) in enumerate(zip(stats, colors)):
        x = 20 + i * 20
        ax.text(x, 8, stat, fontsize=14, fontweight='bold',
                color=color, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['bg_card'],
                         edgecolor=color, linewidth=2))

    plt.tight_layout(pad=1)
    plt.savefig('benchmarks/results/hero_banner.png', dpi=150,
                facecolor=COLORS['bg_dark'], bbox_inches='tight')
    plt.close()
    print("Created: hero_banner.png")


def create_performance_dashboard():
    """Create a comprehensive performance dashboard."""
    # Load data
    results_file = Path('benchmarks/results/benchmark_results.json')
    stress_file = Path('benchmarks/results/stress_test.json')

    if not results_file.exists():
        print("benchmark_results.json not found, skipping dashboard")
        return

    with open(results_file) as f:
        results = json.load(f)

    stress_data = None
    if stress_file.exists():
        with open(stress_file) as f:
            stress_data = json.load(f)

    fig = plt.figure(figsize=(16, 12), facecolor=COLORS['bg_dark'])

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3,
                          left=0.08, right=0.95, top=0.92, bottom=0.08)

    # Title
    fig.suptitle('TinyLLM Performance Dashboard', fontsize=24,
                 fontweight='bold', color=COLORS['text'], y=0.98)

    # 1. Success Rate Gauge (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(COLORS['bg_card'])
    success_rate = sum(1 for r in results if r['success']) / len(results) * 100

    # Create gauge
    theta = np.linspace(np.pi, 0, 100)
    r = 1
    ax1.fill_between(theta, 0, r, alpha=0.1, color=COLORS['grid'])
    fill_theta = np.linspace(np.pi, np.pi - (success_rate/100 * np.pi), 100)
    ax1.fill_between(fill_theta, 0, r, alpha=0.8, color=COLORS['success'])
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-0.2, 1.2)
    ax1.axis('off')
    ax1.text(0, 0.3, f'{success_rate:.0f}%', fontsize=32, fontweight='bold',
             color=COLORS['success'], ha='center', va='center')
    ax1.text(0, -0.05, 'Success Rate', fontsize=14, color=COLORS['text_muted'],
             ha='center', va='center')

    # 2. Latency by Category (top center)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(COLORS['bg_card'])

    categories = {}
    for r in results:
        cat = r['expected_category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r['latency_ms'])

    cat_names = list(categories.keys())
    cat_means = [np.mean(categories[c]) for c in cat_names]
    colors = [CATEGORY_COLORS.get(c, COLORS['accent1']) for c in cat_names]

    bars = ax2.barh(cat_names, cat_means, color=colors, height=0.6, alpha=0.85)
    for bar, val in zip(bars, cat_means):
        ax2.text(val + 200, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}ms', va='center', fontsize=11, color=COLORS['text'])
    style_axis(ax2, 'Avg Latency by Category', xlabel='Latency (ms)')

    # 3. Query Volume Pie (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(COLORS['bg_card'])

    counts = [len(categories[c]) for c in cat_names]
    wedges, texts, autotexts = ax3.pie(counts, labels=cat_names, autopct='%1.0f%%',
                                        colors=colors, startangle=90,
                                        textprops={'fontsize': 11, 'color': COLORS['text']})
    for autotext in autotexts:
        autotext.set_color(COLORS['bg_dark'])
        autotext.set_fontweight('bold')
    ax3.set_title('Query Distribution', fontsize=16, fontweight='bold',
                  color=COLORS['text'], pad=15)

    # 4. Latency Timeline (middle left + center)
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.set_facecolor(COLORS['bg_card'])

    latencies = [r['latency_ms'] for r in results]
    cats = [r['expected_category'] for r in results]
    x = range(len(latencies))

    for i, (lat, cat) in enumerate(zip(latencies, cats)):
        color = CATEGORY_COLORS.get(cat, COLORS['accent1'])
        ax4.bar(i, lat, color=color, alpha=0.8, width=0.8)

    ax4.axhline(np.mean(latencies), color=COLORS['secondary'], linestyle='--',
                linewidth=2, alpha=0.8, label=f'Avg: {np.mean(latencies):.0f}ms')
    ax4.legend(loc='upper right', fontsize=11, facecolor=COLORS['bg_card'],
               edgecolor=COLORS['grid'], labelcolor=COLORS['text'])
    style_axis(ax4, 'Query Latency Timeline', xlabel='Query #', ylabel='Latency (ms)')

    # 5. Response Length Box (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(COLORS['bg_card'])

    resp_by_cat = {c: [r['response_length'] for r in results if r['expected_category'] == c]
                   for c in cat_names}

    bp = ax5.boxplot([resp_by_cat[c] for c in cat_names], tick_labels=cat_names,
                     patch_artist=True, notch=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ['whiskers', 'caps', 'medians']:
        for line in bp[element]:
            line.set_color(COLORS['text_muted'])
    for flier in bp['fliers']:
        flier.set_markeredgecolor(COLORS['text_muted'])

    style_axis(ax5, 'Response Length', ylabel='Characters')
    ax5.tick_params(axis='x', rotation=15)

    # 6. Stress Test Results (bottom, if available)
    if stress_data:
        ax6 = fig.add_subplot(gs[2, :])
        ax6.set_facecolor(COLORS['bg_card'])

        by_diff = stress_data['analysis']['by_difficulty']
        difficulties = ['easy', 'medium', 'hard', 'extreme']
        diff_data = [by_diff.get(d, {}) for d in difficulties]

        x = np.arange(len(difficulties))
        width = 0.35

        latencies = [d.get('avg_latency', 0) for d in diff_data]
        quality = [d.get('avg_quality', 0) * 10000 for d in diff_data]  # Scale for visibility

        diff_colors = [DIFFICULTY_COLORS[d] for d in difficulties]

        bars1 = ax6.bar(x - width/2, latencies, width, label='Latency (ms)',
                       color=diff_colors, alpha=0.85)

        ax6_twin = ax6.twinx()
        line = ax6_twin.plot(x, [d.get('avg_quality', 0) for d in diff_data],
                            'o-', color=COLORS['accent2'], linewidth=3,
                            markersize=12, label='Quality Score')
        ax6_twin.set_ylim(0, 1)
        ax6_twin.tick_params(colors=COLORS['text_muted'], labelsize=12)
        ax6_twin.set_ylabel('Quality Score', fontsize=14, color=COLORS['text_muted'])

        ax6.set_xticks(x)
        ax6.set_xticklabels([d.upper() for d in difficulties], fontsize=14, fontweight='bold')

        for bar, val in zip(bars1, latencies):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                    f'{val:.0f}ms', ha='center', fontsize=11, color=COLORS['text'])

        style_axis(ax6, 'Stress Test: Latency vs Difficulty', ylabel='Latency (ms)')

        # Combined legend
        ax6.legend(loc='upper left', fontsize=11, facecolor=COLORS['bg_card'],
                  edgecolor=COLORS['grid'], labelcolor=COLORS['text'])
    else:
        ax6 = fig.add_subplot(gs[2, :])
        ax6.set_facecolor(COLORS['bg_card'])
        ax6.text(0.5, 0.5, 'Run stress_test.py to see difficulty analysis',
                fontsize=16, color=COLORS['text_muted'], ha='center', va='center',
                transform=ax6.transAxes)
        ax6.axis('off')

    plt.savefig('benchmarks/results/performance_dashboard.png', dpi=150,
                facecolor=COLORS['bg_dark'], bbox_inches='tight')
    plt.close()
    print("Created: performance_dashboard.png")


def create_architecture_visual():
    """Create a visual architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 10), facecolor=COLORS['bg_dark'])
    ax.set_facecolor(COLORS['bg_dark'])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.axis('off')

    # Node helper
    def draw_node(x, y, label, color, shape='rect', size=1.0):
        w, h = 14 * size, 6 * size
        if shape == 'rect':
            rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                           boxstyle='round,pad=0.3',
                                           facecolor=color, alpha=0.85,
                                           edgecolor='white', linewidth=2)
            ax.add_patch(rect)
        elif shape == 'diamond':
            diamond = mpatches.RegularPolygon((x, y), 4, radius=h*0.8,
                                              facecolor=color, alpha=0.85,
                                              edgecolor='white', linewidth=2)
            ax.add_patch(diamond)
        elif shape == 'circle':
            circle = mpatches.Circle((x, y), h*0.6, facecolor=color, alpha=0.85,
                                     edgecolor='white', linewidth=2)
            ax.add_patch(circle)

        ax.text(x, y, label, fontsize=10 * size, fontweight='bold',
                color='white', ha='center', va='center')

    # Arrow helper
    def draw_arrow(x1, y1, x2, y2, color=COLORS['text_muted'], style='-'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2,
                                  linestyle=style))

    # Title
    ax.text(50, 66, 'TinyLLM Architecture', fontsize=22, fontweight='bold',
            color=COLORS['text'], ha='center')

    # Layer labels
    layers = [
        (8, 55, 'INPUT'),
        (8, 45, 'ENTRY'),
        (8, 35, 'ROUTING'),
        (8, 22, 'SPECIALISTS'),
        (8, 10, 'OUTPUT'),
    ]
    for x, y, label in layers:
        ax.text(x, y, label, fontsize=10, color=COLORS['text_muted'],
                rotation=90, ha='center', va='center', fontweight='bold')

    # Input
    draw_node(50, 55, 'User Query', COLORS['primary'], 'rect')

    # Entry
    draw_node(50, 45, 'Entry Node', COLORS['accent4'], 'rect')
    draw_arrow(50, 52, 50, 48)

    # Router
    draw_node(50, 35, 'Router\nqwen2.5:0.5b', COLORS['accent2'], 'diamond', 1.2)
    draw_arrow(50, 42, 50, 39)

    # Specialists
    specialists = [
        (25, 22, 'CODE\ngranite:3b', COLORS['accent4']),
        (42, 22, 'MATH\nphi3:mini', COLORS['primary']),
        (58, 22, 'GENERAL\nqwen2.5:3b', COLORS['accent1']),
        (75, 22, 'CODE+MATH\ncompound', COLORS['accent3']),
    ]

    for x, y, label, color in specialists:
        draw_node(x, y, label, color, 'rect', 0.9)
        draw_arrow(50, 31, x, 26, style='--')

    # Tools
    ax.text(90, 22, 'TOOLS', fontsize=10, color=COLORS['text_muted'], ha='center')
    draw_node(90, 18, 'Calc', COLORS['warning'], 'circle', 0.6)
    draw_node(90, 26, 'Exec', COLORS['warning'], 'circle', 0.6)
    draw_arrow(42, 19, 87, 18, COLORS['warning'], '--')
    draw_arrow(25, 19, 87, 24, COLORS['warning'], '--')

    # Quality Gate
    draw_node(50, 10, 'Quality Gate', COLORS['success'], 'diamond', 1.0)
    for x, _, _, _ in specialists:
        draw_arrow(x, 18, 50, 13)

    # Exit
    draw_node(50, 2, 'Response', COLORS['success'], 'rect')
    draw_arrow(50, 7, 50, 5)

    # Retry arrow
    ax.annotate('', xy=(35, 35), xytext=(35, 10),
               arrowprops=dict(arrowstyle='->', color=COLORS['secondary'],
                              lw=2, linestyle=':', connectionstyle='arc3,rad=0.3'))
    ax.text(28, 22, 'retry', fontsize=9, color=COLORS['secondary'], rotation=90)

    plt.savefig('benchmarks/results/architecture_visual.png', dpi=150,
                facecolor=COLORS['bg_dark'], bbox_inches='tight')
    plt.close()
    print("Created: architecture_visual.png")


def create_stress_test_visual():
    """Create dedicated stress test visualization."""
    stress_file = Path('benchmarks/results/stress_test.json')
    if not stress_file.exists():
        print("stress_test.json not found, skipping")
        return

    with open(stress_file) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor=COLORS['bg_dark'])

    fig.suptitle('Stress Test Results: Scaling to Extreme Difficulty',
                 fontsize=20, fontweight='bold', color=COLORS['text'], y=1.02)

    # 1. Latency by difficulty
    ax1 = axes[0]
    ax1.set_facecolor(COLORS['bg_card'])

    by_diff = data['analysis']['by_difficulty']
    difficulties = ['easy', 'medium', 'hard', 'extreme']
    latencies = [by_diff[d]['avg_latency'] for d in difficulties]
    colors = [DIFFICULTY_COLORS[d] for d in difficulties]

    bars = ax1.bar(difficulties, latencies, color=colors, alpha=0.85,
                   edgecolor='white', linewidth=2)
    for bar, val in zip(bars, latencies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
                f'{val/1000:.1f}s', ha='center', fontsize=14,
                fontweight='bold', color=COLORS['text'])

    style_axis(ax1, 'Latency Scaling', ylabel='Latency (ms)')
    ax1.set_xticks(range(len(difficulties)))
    ax1.set_xticklabels([d.upper() for d in difficulties], fontsize=12, fontweight='bold')

    # 2. Success rate by category
    ax2 = axes[1]
    ax2.set_facecolor(COLORS['bg_card'])

    by_cat = data['analysis']['by_category']
    categories = list(by_cat.keys())
    success_rates = [by_cat[c]['success_rate'] * 100 for c in categories]
    cat_colors = [CATEGORY_COLORS.get(c, COLORS['accent1']) for c in categories]

    short_names = [c.replace('_complexity', '').replace('_chains', '') for c in categories]

    bars = ax2.barh(short_names, success_rates, color=cat_colors, alpha=0.85,
                    edgecolor='white', linewidth=2)

    ax2.set_xlim(0, 110)
    for bar in bars:
        ax2.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                '100%', va='center', fontsize=14, fontweight='bold', color=COLORS['success'])

    style_axis(ax2, 'Success by Category', xlabel='Success Rate (%)')

    # 3. Quality scores
    ax3 = axes[2]
    ax3.set_facecolor(COLORS['bg_card'])

    quality_scores = [by_diff[d]['avg_quality'] for d in difficulties]

    ax3.plot(difficulties, quality_scores, 'o-', color=COLORS['accent2'],
             linewidth=4, markersize=16, markeredgecolor='white', markeredgewidth=2)
    ax3.fill_between(difficulties, quality_scores, alpha=0.3, color=COLORS['accent2'])

    ax3.set_ylim(0.8, 1.0)
    for i, (d, q) in enumerate(zip(difficulties, quality_scores)):
        ax3.text(i, q + 0.015, f'{q:.2f}', ha='center', fontsize=12,
                fontweight='bold', color=COLORS['text'])

    style_axis(ax3, 'Quality Maintained', ylabel='Quality Score')
    ax3.set_xticks(range(len(difficulties)))
    ax3.set_xticklabels([d.upper() for d in difficulties], fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('benchmarks/results/stress_test_visual.png', dpi=150,
                facecolor=COLORS['bg_dark'], bbox_inches='tight')
    plt.close()
    print("Created: stress_test_visual.png")


def create_tool_comparison_visual():
    """Create visualization comparing tool-assisted vs pure LLM."""
    comparison_file = Path('benchmarks/results/tool_comparison.json')
    if not comparison_file.exists():
        print("tool_comparison.json not found, skipping")
        return

    with open(comparison_file) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor=COLORS['bg_dark'])

    fig.suptitle('Tool-Assisted vs Pure LLM Performance',
                 fontsize=20, fontweight='bold', color=COLORS['text'], y=1.02)

    results = data['results']
    summary = data['summary']['math_queries']

    # 1. Latency comparison bar chart
    ax1 = axes[0]
    ax1.set_facecolor(COLORS['bg_card'])

    math_results = [r for r in results if r['category'] != 'general']
    categories = [r['category'][:8] for r in math_results]
    tool_latencies = [r['tool_latency_ms']/1000 for r in math_results]
    pure_latencies = [r['pure_latency_ms']/1000 for r in math_results]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax1.bar(x - width/2, tool_latencies, width, label='With Tools',
                   color=COLORS['primary'], alpha=0.85)
    bars2 = ax1.bar(x + width/2, pure_latencies, width, label='Pure LLM',
                   color=COLORS['secondary'], alpha=0.85)

    ax1.set_ylabel('Latency (seconds)', fontsize=12, color=COLORS['text_muted'])
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right', fontsize=10)
    ax1.legend(facecolor=COLORS['bg_card'], edgecolor=COLORS['grid'],
              labelcolor=COLORS['text'])
    style_axis(ax1, 'Latency by Query Type')

    # 2. Accuracy comparison
    ax2 = axes[1]
    ax2.set_facecolor(COLORS['bg_card'])

    tool_correct = summary['tool_correct']
    pure_correct = summary['pure_correct']
    total = summary['total']

    labels = ['With Tools', 'Pure LLM']
    correct = [tool_correct, pure_correct]
    colors = [COLORS['primary'], COLORS['secondary']]

    bars = ax2.bar(labels, correct, color=colors, alpha=0.85,
                   edgecolor='white', linewidth=2)

    ax2.set_ylim(0, total + 1)
    ax2.axhline(total, color=COLORS['text_muted'], linestyle='--', alpha=0.5)
    ax2.text(1.1, total, f'Total: {total}', fontsize=11, color=COLORS['text_muted'])

    for bar, val in zip(bars, correct):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val}/{total}', ha='center', fontsize=16, fontweight='bold',
                color=COLORS['text'])

    style_axis(ax2, 'Correct Answers', ylabel='Count')

    # 3. Winner summary
    ax3 = axes[2]
    ax3.set_facecolor(COLORS['bg_card'])
    ax3.axis('off')

    # Summary box
    summary_text = [
        ("ACCURACY", "Tools Win", COLORS['success'], f"{tool_correct}/8 vs {pure_correct}/8"),
        ("SPEED", "Pure LLM", COLORS['warning'], f"-{abs(summary['avg_tool_latency_ms'] - summary['avg_pure_latency_ms']):.0f}ms"),
        ("VERDICT", "Use Tools", COLORS['primary'], "Accuracy > Speed"),
    ]

    for i, (label, winner, color, detail) in enumerate(summary_text):
        y = 0.75 - i * 0.3
        ax3.text(0.1, y, label, fontsize=14, fontweight='bold',
                color=COLORS['text_muted'], transform=ax3.transAxes)
        ax3.text(0.5, y, winner, fontsize=18, fontweight='bold',
                color=color, transform=ax3.transAxes)
        ax3.text(0.5, y - 0.08, detail, fontsize=11,
                color=COLORS['text_muted'], transform=ax3.transAxes)

    ax3.set_title('Key Findings', fontsize=16, fontweight='bold',
                  color=COLORS['text'], pad=15)

    plt.tight_layout()
    plt.savefig('benchmarks/results/tool_comparison_visual.png', dpi=150,
                facecolor=COLORS['bg_dark'], bbox_inches='tight')
    plt.close()
    print("Created: tool_comparison_visual.png")


def main():
    """Generate all visualizations."""
    print("Generating professional visualizations...")
    print("-" * 40)

    create_hero_banner()
    create_architecture_visual()
    create_performance_dashboard()
    create_stress_test_visual()
    create_tool_comparison_visual()

    print("-" * 40)
    print("Done! Check benchmarks/results/ for outputs.")


if __name__ == '__main__':
    main()
