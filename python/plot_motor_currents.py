###############################################################################
# plot_motor_currents.py
# Plots motor currents from a CSV file with different line styles and colors
# based on motor type and module number
#
# Line styles:
#   - Front leg motors: dotted lines
#   - Back leg motors: dashed lines with longer segments
#   - Body motors: solid lines
#
# Colors: Different for each module
###############################################################################

import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys

# --- Configuration ----------------------------------------------------------

# Line style definitions
LINE_STYLES = {
    'front_leg': (0, (1, 1)),      # dotted (small dashes and gaps)
    'back_leg': (0, (5, 5)),       # dashed (longer dashes and gaps)
    'body': '-'                    # solid
}

# Color palette for different modules
COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
]

# --- Functions ---------------------------------------------------------------

def load_csv_data(csv_filename):
    """Load motor current data from CSV file."""
    data = {}
    
    try:
        with open(csv_filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                timestamp = float(row['timestamp_ms'])
                module_id = int(row['module_id'])
                motor_type = row['motor_type']
                motor_id = int(row['motor_id'])
                current_ma = float(row['current_ma'])
                
                # Create key for this motor
                key = (module_id, motor_type, motor_id)
                
                if key not in data:
                    data[key] = {'timestamps': [], 'currents': []}
                
                data[key]['timestamps'].append(timestamp)
                data[key]['currents'].append(current_ma)
        
        return data
    
    except FileNotFoundError:
        print(f"Error: File '{csv_filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)

def plot_motor_currents(csv_filename, output_filename=None):
    """
    Plot motor currents from CSV with different line styles and colors.
    """
    
    # Load data
    data = load_csv_data(csv_filename)
    
    if not data:
        print("Error: No data found in CSV file.")
        sys.exit(1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get unique modules and sort
    modules = sorted(set(key[0] for key in data.keys()))
    
    # Plot each motor's data
    for key in sorted(data.keys()):
        module_id, motor_type, motor_id = key
        timestamps = data[key]['timestamps']
        currents = data[key]['currents']
        
        # Get color for this module
        color = COLORS[module_id % len(COLORS)]
        
        # Get line style for this motor type
        linestyle = LINE_STYLES[motor_type]
        
        # Create label
        label = f"Module {module_id} - {motor_type} (ID:{motor_id})"
        
        # Plot the data
        ax.plot(timestamps, currents, 
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=2,
                marker='',
                alpha=0.8)
    
    # Customize plot
    ax.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Current (mA)', fontsize=12, fontweight='bold')
    ax.set_title('Motor Current Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=9, ncol=2)
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show
    if output_filename:
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_filename}")
    else:
        # Generate default filename
        csv_path = Path(csv_filename)
        output_filename = csv_path.stem + "_plot.png"
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_filename}")
    
    # Show plot
    plt.show()

def create_legend_reference():
    """Create a standalone legend/reference image."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create legend elements for motor types
    motor_type_patches = [
        mpatches.Patch(color='black', linestyle=(0, (1, 1)), 
                       label='Front Leg (Dotted)', linewidth=2),
        mpatches.Patch(color='black', linestyle=(0, (5, 5)), 
                       label='Back Leg (Dashed)', linewidth=2),
        mpatches.Patch(color='black', linestyle='-', 
                       label='Body (Solid)', linewidth=2),
    ]
    
    # Create legend elements for modules
    module_patches = []
    for i, color in enumerate(COLORS[:5]):  # Show first 5 modules as example
        module_patches.append(mpatches.Patch(color=color, label=f'Module {i}'))
    
    # Create text for the plot
    ax.text(0.5, 0.85, 'Motor Current Plot Legend', 
            ha='center', fontsize=16, fontweight='bold', transform=ax.transAxes)
    
    ax.text(0.5, 0.70, 'Motor Types:', 
            ha='center', fontsize=12, fontweight='bold', transform=ax.transAxes)
    
    # Add motor type legend
    motor_legend = ax.legend(handles=motor_type_patches, loc='upper center',
                            bbox_to_anchor=(0.5, 0.65), fontsize=11)
    ax.add_artist(motor_legend)
    
    ax.text(0.5, 0.50, 'Module Colors:', 
            ha='center', fontsize=12, fontweight='bold', transform=ax.transAxes)
    
    # Add module legend
    module_legend = ax.legend(handles=module_patches, loc='upper center',
                             bbox_to_anchor=(0.5, 0.45), fontsize=11)
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('motor_currents_legend.png', dpi=150, bbox_inches='tight')
    print("Legend reference saved to: motor_currents_legend.png")
    plt.show()

# --- Entry point ---

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_motor_currents.py <csv_filename> [output_filename]")
        print("\nExample:")
        print("  python plot_motor_currents.py motor_currents_20240101_120000.csv")
        print("  python plot_motor_currents.py motor_currents_20240101_120000.csv output_plot.png")
        print("\nTo create a legend reference, use:")
        print("  python plot_motor_currents.py --legend")
        sys.exit(1)
    
    if sys.argv[1] == '--legend':
        create_legend_reference()
    else:
        csv_filename = sys.argv[1]
        output_filename = sys.argv[2] if len(sys.argv) > 2 else None
        plot_motor_currents(csv_filename, output_filename)
