import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def actual_vs_predicted_plot(y_true, y_pred):
    """
    Interactive scatter plot for Actual vs Predicted values using Plotly
    """
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(
            size=8,
            color='#6366f1',
            opacity=0.6
        )
    ))
    
    # Perfect prediction line
    min_val = min(y_true. min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    
    fig.add_trace(go. Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='#ec4899', dash='dash', width=2)
    ))
    
    fig.update_layout(
        title='Actual vs Predicted Values',
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f1f5f9'),
        hovermode='closest'
    )
    
    return fig

def residual_plot(y_true, y_pred):
    """
    Residual plot to check model assumptions
    """
    residuals = y_true - y_pred
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        marker=dict(
            size=8,
            color='#8b5cf6',
            opacity=0.6
        )
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="#ec4899", line_width=2)
    
    fig.update_layout(
        title='Residual Plot',
        xaxis_title='Predicted Values',
        yaxis_title='Residuals',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f1f5f9')
    )
    
    return fig

def error_distribution_plot(y_true, y_pred):
    """
    Distribution of prediction errors
    """
    errors = y_true - y_pred
    
    fig = px.histogram(
        x=errors,
        nbins=30,
        title='Distribution of Prediction Errors',
        labels={'x': 'Error', 'y': 'Frequency'},
        color_discrete_sequence=['#6366f1']
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f1f5f9'),
        showlegend=False
    )
    
    return fig

def feature_importance_plot(model, feature_names):
    """
    Bar plot showing feature importance (coefficients for linear regression)
    """
    importance = np.abs(model.coef_)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')
    
    bars = ax.barh(feature_names, importance, color='#6366f1')
    ax.set_xlabel('Absolute Coefficient Value', color='#f1f5f9')
    ax.set_title('Feature Importance', color='#f1f5f9', fontweight='bold')
    ax.tick_params(colors='#f1f5f9')
    ax.spines['bottom'].set_color('#475569')
    ax.spines['left'].set_color('#475569')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig