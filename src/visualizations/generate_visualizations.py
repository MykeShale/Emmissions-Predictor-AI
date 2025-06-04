import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import os

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with num_vars axes."""
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(theta, *args, closed=closed, **kwargs)

        def plot(self, *args, **kwargs):
            return super().plot(theta, *args, **kwargs)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

    register_projection(RadarAxes)
    return theta

def create_impact_diagram():
    """Create a diagram showing the impact of the project."""
    # Create output directory if it doesn't exist
    os.makedirs('pitch_deck_assets', exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create a radial diagram
    categories = ['Technical\nInnovation', 'Environmental\nImpact', 
                 'Policy\nRelevance', 'Educational\nValue', 'Scalability']
    values = [0.9, 0.85, 0.8, 0.75, 0.9]
    
    theta = radar_factory(len(categories))
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='radar'))
    ax.plot(theta, values)
    ax.fill(theta, values, alpha=0.25)
    ax.set_varlabels(categories)
    ax.set_title('Project Impact Assessment', pad=20)
    
    plt.savefig('pitch_deck_assets/impact_diagram.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_workflow_diagram():
    """Create a diagram showing the project workflow."""
    plt.figure(figsize=(12, 6))
    
    # Create a horizontal flowchart
    steps = ['Data\nCollection', 'Data\nPreprocessing', 'Model\nTraining', 
             'Model\nEvaluation', 'Results\nVisualization']
    
    # Create boxes
    for i, step in enumerate(steps):
        plt.text(i, 0, step, ha='center', va='center', 
                bbox=dict(facecolor='lightblue', alpha=0.5, boxstyle='round,pad=0.5'))
        
        # Add arrows between boxes
        if i < len(steps) - 1:
            plt.arrow(i + 0.5, 0, 0.5, 0, head_width=0.1, head_length=0.1, 
                     fc='gray', ec='gray')
    
    plt.xlim(-0.5, len(steps) - 0.5)
    plt.ylim(-1, 1)
    plt.axis('off')
    plt.title('Project Workflow')
    
    plt.savefig('pitch_deck_assets/workflow_diagram.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_feature_importance():
    """Create a feature importance visualization."""
    plt.figure(figsize=(10, 6))
    
    # Sample feature importance data
    features = ['GDP per capita', 'Energy consumption', 'Population']
    importance = [0.45, 0.35, 0.20]
    
    # Create horizontal bar chart
    plt.barh(features, importance)
    plt.xlabel('Importance Score')
    plt.title('Feature Importance in CO2 Emissions Prediction')
    
    plt.savefig('pitch_deck_assets/feature_importance.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_emissions_trend():
    """Create a CO2 emissions trend visualization."""
    plt.figure(figsize=(10, 6))
    
    # Sample data
    years = np.arange(1990, 2021)
    emissions = np.random.normal(0, 1, len(years)).cumsum() + 100
    
    plt.plot(years, emissions, 'b-', linewidth=2)
    plt.fill_between(years, emissions, alpha=0.2)
    plt.xlabel('Year')
    plt.ylabel('CO2 Emissions (metric tons)')
    plt.title('Global CO2 Emissions Trend')
    plt.grid(True, alpha=0.3)
    
    plt.savefig('pitch_deck_assets/emissions_trend.png', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    """Generate all visualizations."""
    create_impact_diagram()
    create_workflow_diagram()
    create_feature_importance()
    create_emissions_trend()
    print("All visualizations have been generated in the 'pitch_deck_assets' directory.")

if __name__ == "__main__":
    main() 