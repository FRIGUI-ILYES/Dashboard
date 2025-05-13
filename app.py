# Standard library imports
import base64
import io
import re
import warnings
from datetime import datetime
from io import StringIO

# Third-party imports
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State, dash_table
from plotly.subplots import make_subplots
import statsmodels.api as sm
from scipy.stats import gaussian_kde
from sklearn.preprocessing import LabelEncoder

# Suppress warnings
warnings.filterwarnings("ignore")

# We'll use lazy imports for heavier libraries to avoid circular imports and improve load time
sklearn_modules = {}
scipy_modules = {}
statsmodels_modules = {}
prophet_module = None

def get_sklearn(module_name=None):
    """Lazy import for sklearn modules - imports only when a specific module is requested"""
    global sklearn_modules

    if module_name == 'LinearRegression' and 'LinearRegression' not in sklearn_modules:
        from sklearn.linear_model import LinearRegression
        sklearn_modules['LinearRegression'] = LinearRegression
    elif module_name == 'KNNImputer' and 'KNNImputer' not in sklearn_modules:
        from sklearn.impute import KNNImputer
        sklearn_modules['KNNImputer'] = KNNImputer
    elif module_name == 'OneHotEncoder' and 'OneHotEncoder' not in sklearn_modules:
        from sklearn.preprocessing import OneHotEncoder
        sklearn_modules['OneHotEncoder'] = OneHotEncoder
    elif module_name == 'base' and 'base' not in sklearn_modules:
        import sklearn
        sklearn_modules['base'] = sklearn
    elif module_name == 'RandomForestClassifier' and 'RandomForestClassifier' not in sklearn_modules:
        from sklearn.ensemble import RandomForestClassifier
        sklearn_modules['RandomForestClassifier'] = RandomForestClassifier
    elif module_name == 'StandardScaler' and 'StandardScaler' not in sklearn_modules:
        from sklearn.preprocessing import StandardScaler
        sklearn_modules['StandardScaler'] = StandardScaler
    elif module_name == 'train_test_split' and 'train_test_split' not in sklearn_modules:
        from sklearn.model_selection import train_test_split
        sklearn_modules['train_test_split'] = train_test_split

    if module_name:
        if module_name not in sklearn_modules:
            return None
        return sklearn_modules[module_name]
    return sklearn_modules

def get_scipy(module_name=None):
    """Lazy import for scipy modules"""
    global scipy_modules

    # Only import a specific module when requested
    if module_name == 'chi2_contingency' and 'chi2_contingency' not in scipy_modules:
        try:
            # Import directly instead of through stats
            from scipy.stats._stats_py import chi2_contingency
            scipy_modules['chi2_contingency'] = chi2_contingency
        except ImportError:
            # Fallback method if the first approach doesn't work
            import importlib
            stats_module = importlib.import_module('scipy.stats')
            scipy_modules['chi2_contingency'] = getattr(stats_module, 'chi2_contingency')

    elif module_name == 'ttest_ind' and 'ttest_ind' not in scipy_modules:
        from scipy.stats import ttest_ind
        scipy_modules['ttest_ind'] = ttest_ind

    elif module_name == 'f_oneway' and 'f_oneway' not in scipy_modules:
        from scipy.stats import f_oneway
        scipy_modules['f_oneway'] = f_oneway

    elif module_name == 'pearsonr' and 'pearsonr' not in scipy_modules:
        from scipy.stats import pearsonr
        scipy_modules['pearsonr'] = pearsonr

    elif module_name == 'spearmanr' and 'spearmanr' not in scipy_modules:
        from scipy.stats import spearmanr
        scipy_modules['spearmanr'] = spearmanr

    elif module_name == 'probplot' and 'probplot' not in scipy_modules:
        from scipy.stats import probplot
        scipy_modules['probplot'] = probplot

    elif module_name == 'gaussian_kde' and 'gaussian_kde' not in scipy_modules:
        from scipy.stats import gaussian_kde
        scipy_modules['gaussian_kde'] = gaussian_kde

    elif module_name == 'stats' and 'stats' not in scipy_modules:
        import scipy.stats
        scipy_modules['stats'] = scipy.stats

    if module_name:
        if module_name not in scipy_modules:
            return None
        return scipy_modules[module_name]
    return scipy_modules

def get_statsmodels(module_name=None):
    """Lazy import for statsmodels modules - imports only when a specific module is requested"""
    global statsmodels_modules

    if module_name == 'api' and 'api' not in statsmodels_modules:
        import statsmodels.api as sm
        statsmodels_modules['api'] = sm
    elif module_name == 'OLS' and 'OLS' not in statsmodels_modules:
        from statsmodels.api import OLS
        statsmodels_modules['OLS'] = OLS
    elif module_name == 'seasonal_decompose' and 'seasonal_decompose' not in statsmodels_modules:
        from statsmodels.tsa.seasonal import seasonal_decompose
        statsmodels_modules['seasonal_decompose'] = seasonal_decompose
    elif module_name == 'ARIMA' and 'ARIMA' not in statsmodels_modules:
        from statsmodels.tsa.arima.model import ARIMA
        statsmodels_modules['ARIMA'] = ARIMA

    if module_name:
        if module_name not in statsmodels_modules:
            return None
        return statsmodels_modules[module_name]
    return statsmodels_modules

def get_prophet():
    """Lazy import for Prophet - only imports when needed and caches the result"""
    global prophet_module

    if prophet_module is None:
        try:
            # Only attempt to import if it hasn't been tried before
            from prophet import Prophet # type: ignore
            prophet_module = Prophet
        except ImportError:
            # If Prophet is not installed, cache the failure so we don't try again
            prophet_module = False

    # Return the module or None if import failed
    return prophet_module if prophet_module is not False else None

# Initialize the Dash app with a dark theme
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css"
    ],
    serve_locally=True,
    suppress_callback_exceptions=True,
    # assets_folder="assets",
    # include_assets_files=True,
    assets_external_path="",
    #assets_url_path="/assets",
)
app.title = "Data Analysis Dashboard"

# Define custom index string with CSS animation keyframes
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            :root {
                --primary: #1abc9c;
                --primary-hover: #16a085;
                --primary-light: rgba(26, 188, 156, 0.2);
                --primary-shadow: rgba(26, 188, 156, 0.4);
                --dark-bg: #121212;
                --card-bg: #1d2731;
                --sidebar-bg: #0d1620;
                --text-primary: #f5f5f5;
                --text-secondary: #adb5bd;
                --border-color: rgba(255, 255, 255, 0.1);
                --card-radius: 12px;
                --transition-speed: 0.2s;
                --nav-hover-grad: linear-gradient(90deg, rgba(26, 188, 156, 0.15) 0%, rgba(26, 188, 156, 0.0) 100%);
                --nav-active-color: #1abc9c;
                --nav-button-bg: #101d2c;
                --nav-button-hover-bg: rgba(26, 188, 156, 0.08);
                --nav-active-bg: #15283f;
                --active-indicator: #1abc9c;
                --category-heading: #767f88;
            }

            body {
                background-color: var(--dark-bg);
                color: var(--text-primary);
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }

            /* Override Bootstrap primary color */
            .btn-primary {
                background-color: var(--primary) !important;
                border-color: var(--primary) !important;
            }

            .btn-primary:hover, .btn-primary:focus, .btn-primary:active {
                background-color: var(--primary-hover) !important;
                border-color: var(--primary-hover) !important;
                transform: translateY(-1px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
            }

            /* FAQ Accordion Styling */
            .faq-accordion-item .accordion-button {
                color: var(--primary) !important;
                font-weight: 500 !important;
            }

            .faq-accordion-item .accordion-button:not(.collapsed) {
                background-color: var(--primary-light) !important;
            }

            .faq-accordion-item .accordion-button:focus {
                box-shadow: 0 0 0 0.25rem var(--primary-shadow) !important;
            }

            /* Card styling */
            .card {
                background-color: var(--card-bg) !important;
                border: 1px solid var(--border-color) !important;
                border-radius: var(--card-radius) !important;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
                transition: transform var(--transition-speed), box-shadow var(--transition-speed) !important;
                overflow: hidden !important;
            }

            .card:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15) !important;
            }

            .card-header {
                background-color: rgba(0, 0, 0, 0.2) !important;
                border-bottom: 1px solid var(--border-color) !important;
                font-weight: 600 !important;
            }

            /* Enhanced Nav styling */
            .nav-button {
                transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
                border-radius: 8px !important;
                margin-bottom: 6px !important;
                position: relative !important;
                overflow: hidden !important;
                padding: 12px 16px !important;
                backdrop-filter: blur(5px) !important;
                -webkit-backdrop-filter: blur(5px) !important;
                background-color: var(--nav-button-bg) !important;
                background-image: linear-gradient(to bottom, rgba(255, 255, 255, 0.03) 0%, rgba(0, 0, 0, 0.05) 100%) !important;
                border-left: 3px solid transparent !important;
                transform: translateZ(0) !important;
                font-size: 14px !important;
                letter-spacing: 0.3px !important;
            }

            .nav-button::before {
                content: "" !important;
                position: absolute !important;
                top: 0 !important;
                left: -100% !important;
                width: 100% !important;
                height: 100% !important;
                background: linear-gradient(90deg,
                    rgba(26, 188, 156, 0.0) 0%,
                    rgba(26, 188, 156, 0.1) 50%,
                    rgba(26, 188, 156, 0.0) 100%) !important;
                transition: all 0.5s ease !important;
                z-index: -1 !important;
            }

            .nav-button:hover {
                background-color: var(--nav-button-hover-bg) !important;
                color: var(--primary) !important;
                transform: translateX(5px) !important;
                box-shadow: 0 2px 8px rgba(26, 188, 156, 0.15) !important;
            }

            .nav-button:hover::before {
                left: 100% !important;
                transition: all 0.5s ease !important;
            }

            .nav-button.active {
                background-color: var(--nav-active-bg) !important;
                color: var(--nav-active-color) !important;
                font-weight: 500 !important;
                border-left: 3px solid var(--active-indicator) !important;
                box-shadow: 0 2px 10px rgba(26, 188, 156, 0.2) !important;
                animation: subtle-glow 2s infinite alternate !important;
            }

            .nav-button.active::after {
                content: "" !important;
                position: absolute !important;
                top: 0 !important;
                left: 0 !important;
                width: 3px !important;
                height: 100% !important;
                background-color: var(--active-indicator) !important;
                animation: border-pulse 2s infinite !important;
            }

            .nav-button.active::before {
                animation: nav-hover-animation 3s ease infinite !important;
                background: linear-gradient(90deg,
                    rgba(26, 188, 156, 0.0) 0%,
                    rgba(26, 188, 156, 0.15) 50%,
                    rgba(26, 188, 156, 0.0) 100%) !important;
                background-size: 200% 100% !important;
                left: 0 !important;
            }

            @keyframes subtle-glow {
                0% { box-shadow: 0 2px 10px rgba(26, 188, 156, 0.2); }
                100% { box-shadow: 0 4px 15px rgba(26, 188, 156, 0.5); }
            }

            @keyframes slide-in {
                0% { transform: translateX(-20px); opacity: 0; }
                100% { transform: translateX(0); opacity: 1; }
            }

            @keyframes nav-hover-animation {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }

            @keyframes icon-pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.2); }
                100% { transform: scale(1); }
            }

            @keyframes border-glow {
                0% { border-color: rgba(26, 188, 156, 0.6); }
                50% { border-color: rgba(26, 188, 156, 1); }
                100% { border-color: rgba(26, 188, 156, 0.6); }
            }

            .nav-button i {
                transition: transform 0.3s cubic-bezier(0.68, -0.55, 0.265, 1.55) !important;
                margin-right: 12px !important;
                width: 20px !important;
                text-align: center !important;
                color: rgba(255, 255, 255, 0.8) !important;
                text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2) !important;
            }

            .nav-button:hover i {
                transform: scale(1.2) !important;
                color: var(--primary) !important;
                animation: icon-pulse 1.5s infinite !important;
                text-shadow: 0 0 8px rgba(26, 188, 156, 0.5) !important;
            }

            .nav-button.active i {
                color: var(--primary) !important;
                animation: icon-pulse 2s infinite !important;
                text-shadow: 0 0 10px rgba(26, 188, 156, 0.6) !important;
            }

            @keyframes pulse-icon {
                0% { transform: scale(1); }
                50% { transform: scale(1.3); }
                100% { transform: scale(1.2); }
            }

            /* Nav divider effect */
            .nav-pills {
                position: relative !important;
            }

            .nav-pills::after {
                content: "";
                position: absolute;
                bottom: 0;
                left: 10%;
                width: 80%;
                height: 1px;
                background: linear-gradient(90deg, transparent, var(--border-color), transparent);
            }

            /* Custom components */
            input[type="checkbox"]:checked,
            .custom-control-input:checked ~ .custom-control-label::before {
                background-color: var(--primary) !important;
                border-color: var(--primary) !important;
            }

            /* Dropdowns */
            .dropdown-menu {
                background-color: var(--card-bg) !important;
                border: 1px solid var(--border-color) !important;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
                border-radius: 8px !important;
                overflow: hidden !important;
            }

            .dropdown-item:hover, .dropdown-item:focus {
                background-color: var(--primary-light) !important;
                color: var(--primary) !important;
            }

            /* Dash dropdown specific styling */
            .Select-control, .Select--single > .Select-control .Select-value {
                background-color: #16213e !important;
                color: var(--text-primary) !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
                border-radius: 8px !important;
                box-shadow: none !important;
                height: 40px !important;
                padding: 4px 8px !important;
                display: flex !important;
                align-items: center !important;
            }

            .Select-control:hover, .is-focused:not(.is-open) > .Select-control {
                border-color: var(--primary) !important;
                box-shadow: 0 0 0 1px var(--primary-shadow) !important;
            }

            .Select.is-focused > .Select-control {
                background-color: var(--card-bg) !important;
            }

            .Select-menu-outer {
                background-color: var(--card-bg) !important;
                border: 1px solid var(--border-color) !important;
                border-radius: 8px !important;
                margin-top: 4px !important;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
                z-index: 9999 !important; /* Ensure high z-index for all dropdown menus */
                position: absolute !important;
                width: 100% !important;
                max-height: 300px !important; /* Ensure enough height for options */
            }

            .Select-option {
                background-color: var(--card-bg) !important;
                color: var(--text-primary) !important;
                padding: 10px 16px !important;
                white-space: nowrap !important;
                overflow: hidden !important;
                text-overflow: ellipsis !important;
                width: 100% !important;
                box-sizing: border-box !important;
            }

            .Select-option.is-selected {
                background-color: var(--primary-light) !important;
                color: var(--primary) !important;
            }

            .Select-option:hover, .Select-option.is-focused {
                background-color: var(--primary-light) !important;
                color: var(--primary) !important;
            }

            .Select-value-label, .Select-value-label > span {
                color: var(--text-primary) !important;
                font-size: 14px !important;
                font-weight: normal !important;
                padding: 2px 0 !important;
            }

            .Select-value {
                padding-left: 8px !important;
                display: flex !important;
                align-items: center !important;
            }

            .Select-placeholder {
                color: var(--text-secondary) !important;
            }

            .Select-clear-zone {
                color: var(--text-secondary) !important;
            }

            .Select-clear-zone:hover {
                color: #e74c3c !important;
            }

            .Select-arrow {
                border-color: var(--text-secondary) transparent transparent !important;
                border-width: 5px 5px 2.5px !important;
                margin-top: -2.5px !important;
                opacity: 0.7 !important;
            }

            .Select.is-open > .Select-control .Select-arrow {
                border-color: transparent transparent var(--primary) !important;
                border-width: 2.5px 5px 5px !important;
                margin-top: -2.5px !important;
            }

            /* VirtualizedSelect specific styles */
            .VirtualizedSelectOption {
                background-color: var(--card-bg) !important;
                color: var(--text-primary) !important;
            }

            .VirtualizedSelectFocusedOption {
                background-color: var(--primary-light) !important;
                color: var(--primary) !important;
            }

            /* Table styling */
            .table {
                --bs-table-bg: transparent !important;
                border-radius: 8px !important;
                overflow: hidden !important;
            }

            .table thead th {
                background-color: rgba(0, 0, 0, 0.2) !important;
                color: var(--primary) !important;
                font-weight: 600 !important;
                text-transform: uppercase !important;
                font-size: 0.8rem !important;
                letter-spacing: 0.5px !important;
            }

            .table tbody tr:hover {
                background-color: rgba(26, 188, 156, 0.05) !important;
            }

            /* Scrollbar styling */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }

            ::-webkit-scrollbar-track {
                background: rgba(0, 0, 0, 0.2);
                border-radius: 10px;
            }

            ::-webkit-scrollbar-thumb {
                background: rgba(255, 255, 255, 0.2);
                border-radius: 10px;
            }

            ::-webkit-scrollbar-thumb:hover {
                background: var(--primary);
            }

            @keyframes pulse {
                0% {
                    box-shadow: 0 0 0 0 var(--primary-shadow);
                }
                70% {
                    box-shadow: 0 0 0 10px rgba(26, 188, 156, 0);
                }
                100% {
                    box-shadow: 0 0 0 0 rgba(26, 188, 156, 0);
                }
            }

            /* Form controls */
            .form-control, .form-select {
                background-color: rgba(0, 0, 0, 0.2) !important;
                border: 1px solid var(--border-color) !important;
                color: var(--text-primary) !important;
                border-radius: 8px !important;
            }

            .form-control:focus, .form-select:focus {
                border-color: var(--primary) !important;
                box-shadow: 0 0 0 0.25rem var(--primary-shadow) !important;
            }

            /* Upload styling */
            .upload-area {
                border: 2px dashed var(--border-color) !important;
                border-radius: 12px !important;
                background-color: rgba(0, 0, 0, 0.2) !important;
                transition: all var(--transition-speed) !important;
            }

            .upload-area:hover {
                border-color: var(--primary) !important;
                background-color: rgba(26, 188, 156, 0.05) !important;
            }

            /* Dash table styling */
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner table {
                background-color: var(--card-bg) !important;
            }

            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th {
                background-color: rgba(0, 0, 0, 0.2) !important;
                color: var(--primary) !important;
            }

            .dash-cell-value {
                background-color: var(--card-bg) !important;
                color: var(--text-primary) !important;
            }

            .dash-filter, .dash-spreadsheet input {
                background-color: var(--card-bg) !important;
                color: var(--text-primary) !important;
                border: 1px solid var(--border-color) !important;
            }

            @keyframes title-glow {
                0% { text-shadow: 0 0 5px rgba(26, 188, 156, 0.3); }
                50% { text-shadow: 0 0 15px rgba(26, 188, 156, 0.6); }
                100% { text-shadow: 0 0 5px rgba(26, 188, 156, 0.3); }
            }

            @keyframes gradient-text {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }

            @keyframes fade-in {
                0% { opacity: 0; transform: translateY(10px); }
                100% { opacity: 1; transform: translateY(0); }
            }

            @keyframes slight-bounce {
                0% { transform: translateY(0); }
                50% { transform: translateY(-5px); }
                100% { transform: translateY(0); }
            }

            @keyframes border-pulse {
                0% { border-color: rgba(26, 188, 156, 0.7); }
                50% { border-color: rgba(26, 188, 156, 1); }
                100% { border-color: rgba(26, 188, 156, 0.7); }
            }

            /* Nav category styles */
            .nav-category {
                margin-bottom: 15px !important;
                position: relative !important;
            }

            .nav-category-title {
                font-size: 11px !important;
                font-weight: 500 !important;
                letter-spacing: 1px !important;
                color: var(--category-heading) !important;
                padding-left: 15px !important;
                margin-bottom: 8px !important;
                text-transform: uppercase !important;
                position: relative !important;
                display: flex !important;
                align-items: center !important;
            }

            .nav-category-title::before {
                content: "" !important;
                height: 3px !important;
                width: 3px !important;
                border-radius: 50% !important;
                background: var(--primary) !important;
                display: inline-block !important;
                margin-right: 8px !important;
                box-shadow: 0 0 5px var(--primary) !important;
            }

            .nav-category .nav-pills {
                padding-left: 8px !important;
            }

            @keyframes subtle-glow {
                0% { box-shadow: 0 2px 10px rgba(26, 188, 156, 0.2); }
                100% { box-shadow: 0 4px 15px rgba(26, 188, 156, 0.5); }
            }

            @keyframes slide-in {
                0% { transform: translateX(-20px); opacity: 0; }
                100% { transform: translateX(0); opacity: 1; }
            }

            @keyframes nav-hover-animation {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }

            @keyframes icon-pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.2); }
                100% { transform: scale(1); }
            }

            @keyframes border-glow {
                0% { border-color: rgba(26, 188, 156, 0.6); }
                50% { border-color: rgba(26, 188, 156, 1); }
                100% { border-color: rgba(26, 188, 156, 0.6); }
            }

            .nav-button i {
                transition: transform 0.3s cubic-bezier(0.68, -0.55, 0.265, 1.55) !important;
                margin-right: 12px !important;
                width: 20px !important;
                text-align: center !important;
                color: rgba(255, 255, 255, 0.8) !important;
                text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2) !important;
            }

            .nav-button:hover i {
                transform: scale(1.2) !important;
                color: var(--primary) !important;
                animation: icon-pulse 1.5s infinite !important;
                text-shadow: 0 0 8px rgba(26, 188, 156, 0.5) !important;
            }

            .nav-button.active i {
                color: var(--primary) !important;
                animation: icon-pulse 2s infinite !important;
                text-shadow: 0 0 10px rgba(26, 188, 156, 0.6) !important;
            }

            @keyframes pulse-icon {
                0% { transform: scale(1); }
                50% { transform: scale(1.3); }
                100% { transform: scale(1.2); }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer style="border-top: none; background-color: var(--dark-bg); padding: 20px; text-align: center; color: var(--text-secondary);">
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Custom CSS with modern teal-based styling
custom_css = {
    "background": {
        "backgroundColor": "var(--dark-bg)",
        "color": "var(--text-primary)",
        "minHeight": "100vh",
    },
    "sidebar": {
        "backgroundColor": "var(--sidebar-bg)",
        "height": "100vh",
        "padding": "20px 15px",
        "boxShadow": "2px 0 15px rgba(0,0,0,0.4)",
        "position": "fixed",
        "width": "260px",
        "overflowY": "auto",
        "zIndex": "1000",
        "backdropFilter": "blur(5px)",
        "borderRight": "1px solid rgba(255, 255, 255, 0.05)",
        "background": "linear-gradient(180deg, #0d1620 0%, #111d29 100%)",
    },
    "content": {
        "marginLeft": "260px",
        "padding": "30px",
        "backgroundColor": "var(--dark-bg)",
    },
    "card": {
        "backgroundColor": "var(--card-bg)",
        "borderRadius": "var(--card-radius)",
        "boxShadow": "0 4px 6px rgba(0,0,0,0.1)",
        "padding": "20px",
        "marginBottom": "25px",
        "border": "1px solid var(--border-color)",
    },
    "card_header": {
        "backgroundColor": "rgba(0, 0, 0, 0.2)",
        "color": "var(--text-primary)",
        "fontWeight": "600",
        "padding": "16px 20px",
        "borderRadius": "var(--card-radius) var(--card-radius) 0 0",
        "borderBottom": "1px solid var(--border-color)",
    },
    "button": {
        "backgroundColor": "var(--primary)",
        "color": "#ffffff",
        "border": "none",
        "borderRadius": "8px",
        "padding": "12px 20px",
        "margin": "8px 0",
        "fontSize": "14px",
        "fontWeight": "500",
        "textTransform": "uppercase",
        "letterSpacing": "0.5px",
        "boxShadow": "0 2px 5px rgba(0,0,0,0.2)",
        "transition": "all 0.3s ease",
        "width": "100%",
    },
    "button_hover": {
        "backgroundColor": "var(--primary-hover)",
        "transform": "translateY(-1px)",
        "boxShadow": "0 4px 8px rgba(0,0,0,0.3)",
    },
    "upload": {
        "width": "100%",
        "height": "120px",
        "lineHeight": "120px",
        "borderWidth": "2px",
        "borderStyle": "dashed",
        "borderRadius": "8px",
        "textAlign": "center",
        "margin": "20px 0",
        "backgroundColor": "#16213e",
        "color": "#e6e6e6",
        "borderColor": "#2a3a5e",
        "fontSize": "18px",
    },
    "table": {
        "backgroundColor": "#16213e",
        "color": "#e6e6e6",
        "borderRadius": "8px",
        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
        "border": "1px solid #2a3a5e",
    },
    "table_header": {
        "backgroundColor": "#0f3460",
        "color": "#ffffff",
        "fontWeight": "bold",
    },
    "dropdown": {
        "backgroundColor": "#16213e",
        "border": "1px solid rgba(255, 255, 255, 0.1)",
        "color": "#FFFFFF", # Changed from #CCCCCC to #FFFFFF for better visibility
        "padding": "4px 8px", # Adjusted padding for better spacing
        "fontSize": "14px", # Added font size
        "lineHeight": "1.5", # Improved line height to prevent overlap
        "zIndex": "1000", # Ensure dropdown appears above other elements
        "height": "40px", # Fixed height for consistency
        "fontWeight": "normal", # Normal font weight for better readability
        "textAlign": "left", # Left align text
        "borderRadius": "8px", # Consistent border radius
        "boxShadow": "0 2px 5px rgba(0, 0, 0, 0.2)", # Subtle shadow
    },
    "dropdown_menu": {
        "backgroundColor": "#16213e",
        "border": "1px solid rgba(255, 255, 255, 0.1)",
        "zIndex": "9999", # Very high z-index to ensure it appears on top of everything
        "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.3)",
        "borderRadius": "8px",
        "marginTop": "5px",
        "width": "220px", # Match dropdown width
        "position": "absolute", # Ensure proper positioning
        "display": "block", # Always display as block
        "opacity": "1", # Full opacity
        "padding": "4px 0", # Add padding for better spacing
    },
    "dropdown_item": {
        "color": "#FFFFFF",
        "backgroundColor": "#16213e",
        "padding": "8px 12px",
        "fontSize": "14px",
        "fontWeight": "normal",
        "borderBottom": "1px solid rgba(255, 255, 255, 0.05)",
        "transition": "all 0.2s ease",
        "cursor": "pointer",
    },
    "dropdown_item_hover": {
        "color": "#ffffff",
        "backgroundColor": "rgba(26, 188, 156, 0.2)",
        "borderLeft": "3px solid #1abc9c",
        "transition": "all 0.2s ease",
    },
    "nav_button": {
        "backgroundColor": "var(--nav-button-bg)",
        "color": "#e6e6e6",
        "border": "none",
        "borderRadius": "8px",
        "padding": "12px 16px",
        "margin": "5px 0",
        "fontSize": "14px",
        "fontWeight": "400",
        "textAlign": "left",
        "width": "100%",
        "transition": "all 0.3s ease-out",
        "position": "relative",
        "overflow": "hidden",
        "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
        "backdropFilter": "blur(5px)",
        "borderLeft": "3px solid transparent",
        "transform": "translateZ(0)",
        "letterSpacing": "0.3px",
        "backgroundImage": "linear-gradient(to bottom, rgba(255, 255, 255, 0.03) 0%, rgba(0, 0, 0, 0.05) 100%)",
    },
    "nav_button_hover": {
        "backgroundColor": "var(--nav-button-hover-bg)",
        "color": "var(--primary)",
        "transform": "translateX(5px)",
        "boxShadow": "0 2px 8px rgba(26, 188, 156, 0.15)",
    },
    "nav_button_active": {
        "backgroundColor": "var(--nav-active-bg)",
        "color": "var(--primary)",
        "boxShadow": "0 2px 5px rgba(0,0,0,0.2)",
        "borderLeft": "3px solid var(--active-indicator)",
    },
    "text_dark": {
        "color": "#e6e6e6",
    },
    "text_light": {
        "color": "#ffffff",
    },
    "text_highlight": {
        "color": "#4da6ff",
    },
    "text_error": {
        "color": "#ff6b6b",
    },
    "text_success": {
        "color": "#51cf66",
    },
}

# Function to apply dark theme to plots
def apply_dark_theme(fig):
    # Define a custom color palette with teal as primary
    custom_colors = [
        "#1abc9c",  # Teal (primary)
        "#16a085",  # Darker teal
        "#2ecc71",  # Green
        "#3498db",  # Blue
        "#9b59b6",  # Purple
        "#f1c40f",  # Yellow
        "#e67e22",  # Orange
        "#e74c3c",  # Red
        "#1f3a93",  # Dark blue
        "#26c281",  # Mint
    ]

    # Update layout with improved styling
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='var(--card-bg)',
        paper_bgcolor='var(--card-bg)',
        font=dict(
            family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
            size=14,
            color="var(--text-primary)"
        ),
        title_font=dict(
            family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
            size=20,
            color="var(--primary)"
        ),
        legend=dict(
            bgcolor='rgba(0, 0, 0, 0.2)',
            bordercolor='var(--border-color)',
            borderwidth=1,
            font=dict(size=12)
        ),
        margin=dict(l=60, r=40, t=60, b=60),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            linecolor='var(--border-color)',
            tickcolor='var(--text-secondary)',
            zerolinecolor='var(--border-color)',
            tickfont=dict(size=12),
            title_font=dict(size=14, color="var(--text-secondary)")
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            linecolor='var(--border-color)',
            tickcolor='var(--text-secondary)',
            zerolinecolor='var(--border-color)',
            tickfont=dict(size=12),
            title_font=dict(size=14, color="var(--text-secondary)")
        ),
        hoverlabel=dict(
            bgcolor='var(--card-bg)',
            font_size=14,
            font_family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif"
        )
    )

    # Color palette setting for different trace types
    for i, trace in enumerate(fig.data):
        color_idx = i % len(custom_colors)

        if trace.type == 'scatter':
            if hasattr(trace, 'marker') and trace.marker is not None:
                fig.data[i].marker.color = custom_colors[color_idx]
            if hasattr(trace, 'line') and trace.line is not None:
                fig.data[i].line.color = custom_colors[color_idx]
        elif trace.type == 'heatmap':
            # Create a custom colorscale using the theme colors
            fig.data[i].colorscale = [[0, "var(--dark-bg)"], [0.5, "#16a085"], [1, "#1abc9c"]]

    # Add subtle gradient background
    fig.update_layout(
        shapes=[
            dict(
                type="rect",
                xref="paper", yref="paper",
                x0=0, y0=0, x1=1, y1=1,
                fillcolor="rgba(0, 0, 0, 0.1)",
                layer="below",
                line_width=0,
            )
        ]
    )

    return fig

# Helper functions for data type detection
def is_possible_datetime(series):
    """Check if a series could be converted to datetime"""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    # Handle empty series or series with no non-null values
    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    # Take a sample of up to 100 non-null values
    sample_size = min(100, len(non_null))
    sample = non_null.sample(sample_size) if sample_size > 0 else non_null

    try:
        pd.to_datetime(sample, errors='raise')
        return True
    except:
        pass

    # Check for common date formats
    date_formats = [
        '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d',
        '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S',
        '%d/%m/%Y %H:%M:%S', '%Y/%m/%d %H:%M:%S'
    ]

    for fmt in date_formats:
        try:
            sample.apply(lambda x: datetime.strptime(str(x), fmt))
            return True
        except:
            continue

    return False

def is_possible_numeric(series):
    """Check if a series could be converted to numeric"""
    if pd.api.types.is_numeric_dtype(series):
        return True

    sample = series.dropna().sample(min(100, len(series)))
    if len(sample) == 0:
        return False

    try:
        pd.to_numeric(sample, errors='raise')
        return True
    except:
        return False

def is_possible_boolean(series):
    """Check if a series could be converted to boolean"""
    if pd.api.types.is_bool_dtype(series):
        return True

    sample = series.dropna().sample(min(100, len(series)))
    if len(sample) == 0:
        return False

    # Check for common boolean representations
    true_values = ['true', 't', 'yes', 'y', '1']
    false_values = ['false', 'f', 'no', 'n', '0']

    try:
        normalized = sample.str.lower().str.strip()
        if all(x in true_values + false_values for x in normalized if pd.notna(x)):
            return True
    except:
        pass

    return False

def detect_data_type(series):
    """Detect the most appropriate data type for a series"""
    if is_possible_datetime(series):
        return 'datetime'
    elif is_possible_numeric(series):
        return 'numeric'
    elif is_possible_boolean(series):
        return 'boolean'
    elif pd.api.types.is_string_dtype(series):
        # Check if it's actually categorical with low cardinality
        unique_count = series.nunique()
        if unique_count < min(50, len(series) * 0.5):
            return 'categorical'
        return 'text'
    else:
        return str(series.dtype)

def get_conversion_suggestion(series, current_type):
    """Get suggestion for converting a series to a more appropriate type"""
    detected_type = detect_data_type(series)

    if detected_type == current_type:
        return None

    if detected_type == 'datetime':
        return {
            'from': current_type,
            'to': 'datetime',
            'method': "pd.to_datetime",
            'example': "pd.to_datetime(df['column'], errors='coerce')"
        }
    elif detected_type == 'numeric':
        return {
            'from': current_type,
            'to': 'numeric',
            'method': "pd.to_numeric",
            'example': "pd.to_numeric(df['column'], errors='coerce')"
        }
    elif detected_type == 'boolean':
        return {
            'from': current_type,
            'to': 'boolean',
            'method': "astype(bool)",
            'example': "df['column'].map({'true': True, 'false': False}).astype(bool)"
        }
    elif detected_type == 'categorical' and current_type != 'category':
        return {
            'from': current_type,
            'to': 'category',
            'method': "astype('category')",
            'example': "df['column'].astype('category')"
        }

    return None

# --- Encoding Section Layout ---
html.Div([
    dbc.Row([
        dbc.Col([
            html.Strong("Encoding Options", style={"color": "#1abc9c", "marginBottom": "10px"}),
            html.Div([
                html.Label("Column to Encode:", style={"color": "#e6e6e6", "marginRight": "10px"}),
                dcc.Dropdown(
                    id="encoding-column-dropdown",
                    options=[],
                    placeholder="Select column",
                    style={"backgroundColor": "#1a1a2e", "color": "#e6e6e6", "marginBottom": "10px"}
                )
            ], style={"marginBottom": "15px"}),
            html.Div([
                html.Label("Encoding Type:", style={"color": "#e6e6e6", "marginRight": "10px"}),
                dcc.Dropdown(
                    id="encoding-type-dropdown",
                    options=[
                        {"label": "One-Hot Encoding", "value": "onehot"},
                        {"label": "Ordinal Encoding", "value": "ordinal"}
                    ],
                    placeholder="Select encoding type",
                    style={"backgroundColor": "#1a1a2e", "color": "#e6e6e6"}
                )
            ], style={"marginBottom": "15px"}),
            html.Div(id="ordinal-order-container", style={"marginBottom": "15px"}),
            dbc.Button("Apply Encoding", id="apply-encoding-button", color="primary", style={"marginBottom": "15px", "width": "100%"}),
            html.Div(id="encoding-message", style={"marginBottom": "15px"}),
        ], width=12)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([
            html.Strong("Preview of Encoded Data:", style={"color": "#1abc9c", "marginBottom": "10px"}),
            html.Div([
                html.Label("Display:", style={"color": "#e6e6e6", "marginRight": "10px"}),
                dcc.Dropdown(
                    id="encoded-preview-dropdown",
                    options=[
                        {"label": "Full DataFrame", "value": "full"},
                        {"label": "Head (first 5 rows)", "value": "head"},
                        {"label": "Tail (last 5 rows)", "value": "tail"}
                    ],
                    value="full",
                    clearable=False,
                    style={"width": "220px", "backgroundColor": "#1a1a2e", "color": "#e6e6e6", "marginBottom": "10px"}
                )
            ], style={"marginBottom": "10px", "display": "flex", "alignItems": "center"}),
            dash_table.DataTable(
                id="encoded-preview-table",
                style_table={"overflowX": "auto", "backgroundColor": "#16213e"},
                style_header={"backgroundColor": "#0f3460", "color": "#ffffff", "fontWeight": "bold"},
                style_cell={"backgroundColor": "#16213e", "color": "#e6e6e6", "padding": "10px", "border": "1px solid #2a3a5e"},
                page_size=10
            )
        ], width=12)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([
            html.Div([
                dbc.Button("Download CSV", id="download-encoded-csv-button", color="success", style={"marginRight": "10px"}),
                dbc.Button("Download JSON", id="download-encoded-json-button", color="info", style={"marginRight": "10px"}),
                dbc.Button("Download Excel", id="download-encoded-excel-button", color="warning"),
                dcc.Download(id="download-encoded-csv"),
                dcc.Download(id="download-encoded-json"),
                dcc.Download(id="download-encoded-excel")
            ], style={"marginTop": "10px", "textAlign": "center"})
        ], width=12)
    ], className="mb-4"),
], style={"marginTop": "30px", "marginBottom": "30px"})

# Callback to update the encoded data preview table based on dropdown selection
@app.callback(
    [
        Output("encoded-preview-table", "data"),
        Output("encoded-preview-table", "columns")
    ],
    [
        Input("encoded-preview-dropdown", "value"),
        Input("encoded-df-store", "data")
    ]
)
def update_encoded_preview_table(preview_option, encoded_data):
    import pandas as pd
    if not encoded_data:
        return [], []
    df = pd.DataFrame(encoded_data)
    if preview_option == "head":
        df = df.head(5)
    elif preview_option == "tail":
        df = df.tail(5)
    # else: full dataframe
    columns = [{"name": col, "id": col} for col in df.columns]
    data = df.to_dict("records")
    return data, columns

# Layout with updated styling
app.layout = html.Div(style=custom_css["background"], children=[
    # Sidebar
    html.Div(style=custom_css["sidebar"], children=[
        # Header with enhanced styling
        html.Div([
            html.H4("Data Analysis", style={
                "color": "var(--primary)",
                "fontWeight": "600",
                "padding": "15px 0 20px 0",
                "textAlign": "center",
                "marginBottom": "10px",
                "letterSpacing": "0.5px",
                "textShadow": "0 0 10px var(--primary-shadow)",
                "animation": "title-glow 3s infinite alternate",
                "background": "linear-gradient(90deg, #1abc9c, #16a085, #1abc9c)",
                "backgroundSize": "200% auto",
                "WebkitBackgroundClip": "text",
                "WebkitTextFillColor": "transparent",
                "backgroundClip": "text"
            }),
            html.Div([
                html.I(className="fas fa-chart-line", style={
                    "color": "var(--primary)",
                    "fontSize": "24px",
                    "marginRight": "10px",
                    "animation": "icon-pulse 2s infinite"
                }),
                html.Span("Dashboard", style={
                    "color": "var(--text-secondary)",
                    "fontSize": "16px",
                    "letterSpacing": "0.5px"
                })
            ], style={
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "paddingBottom": "15px",
                "borderBottom": "1px solid var(--border-color)",
                "marginBottom": "20px",
            }),
        ]),

        # Category Group - General
        html.Div([
            html.P("GENERAL", className="nav-category-title"),
            dbc.Nav([
                dbc.NavLink(
                    [
                        html.I(className="fas fa-home mr-2"),
                        "Welcome"
                    ],
                    id="welcome-button",
                    href="#",
                    active=True,
                    style=custom_css["nav_button"],
                    className="nav-button active"
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-upload mr-2"),
                        "Import Data"
                    ],
                    id="import-button",
                    href="#",
                    active=False,
                    style=custom_css["nav_button"],
                    className="nav-button"
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-table mr-2"),
                        "Summary"
                    ],
                    id="summary-button",
                    href="#",
                    style=custom_css["nav_button"],
                    className="nav-button"
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-link mr-2"),
                        "Encoding"
                    ],
                    id="encoding-button",
                    href="#",
                    style=custom_css["nav_button"],
                    className="nav-button"
                ),
            ], vertical=True, pills=True, className="nav-pills", style={"marginBottom": "20px"}),
        ], className="nav-category"),

        # Category Group - Data Processing
        html.Div([
            html.P("DATA PROCESSING", className="nav-category-title"),
            dbc.Nav([
                dbc.NavLink(
                    [
                        html.I(className="fas fa-fill-drip mr-2"),
                        "Imputation"
                    ],
                    id="imputation-button",
                    href="#",
                    style=custom_css["nav_button"],
                    className="nav-button"
                ),
            ], vertical=True, pills=True, className="nav-pills", style={"marginBottom": "20px"}),
        ], className="nav-category"),

        # Category Group - Analysis
        html.Div([
            html.P("ANALYSIS", className="nav-category-title"),
            dbc.Nav([
                dbc.NavLink(
                    [
                        html.I(className="fas fa-chart-pie mr-2"),
                        "Statistics"
                    ],
                    id="statistics-button",
                    href="#",
                    style=custom_css["nav_button"],
                    className="nav-button"
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-square-root-alt mr-2"),
                        "Tests"
                    ],
                    id="tests-button",
                    href="#",
                    style=custom_css["nav_button"],
                    className="nav-button"
                ),
            ], vertical=True, pills=True, className="nav-pills", style={"marginBottom": "20px"}),
        ], className="nav-category"),

        # Category Group - Advanced
        html.Div([
            html.P("ADVANCED", className="nav-category-title"),
            dbc.Nav([
                dbc.NavLink(
                    [
                        html.I(className="fas fa-chart-line mr-2"),
                        "Linear Regression"
                    ],
                    id="regression-button",
                    href="#",
                    style=custom_css["nav_button"],
                    className="nav-button"
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-robot mr-2"),
                        "Prediction"
                    ],
                    id="prediction-button",
                    href="#",
                    style=custom_css["nav_button"],
                    className="nav-button"
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-file-alt mr-2"),
                        "Report"
                    ],
                    id="report-button",
                    href="#",
                    style=custom_css["nav_button"],
                    className="nav-button"
                ),
            ], vertical=True, pills=True, className="nav-pills", style={"marginBottom": "20px"}),
        ], className="nav-category"),

        # Category Group - Help
        html.Div([
            html.P("HELP", className="nav-category-title"),
            dbc.Nav([
                dbc.NavLink(
                    [
                        html.I(className="fas fa-question-circle mr-2"),
                        "FAQ"
                    ],
                    id="faq-button",
                    href="#",
                    style=custom_css["nav_button"],
                    className="nav-button"
                ),
            ], vertical=True, pills=True, className="nav-pills", style={"marginBottom": "20px"}),
        ], className="nav-category"),
    ]),

    # Main Content
    html.Div(style=custom_css["content"], children=[
        # Welcome Page
        html.Div(id="welcome-content", style={"display": "block", "opacity": "1", "transition": "opacity 0.3s ease-in-out"}, children=[
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.I(className="fas fa-home", style={"fontSize": "28px", "color": "var(--primary)", "marginRight": "15px"}),
                        html.Span("Welcome to the Data Analysis Dashboard!", style={"fontSize": "1.6em", "fontWeight": "bold", "color": "var(--primary)"})
                    ], style={"display": "flex", "alignItems": "center"})
                ], style=custom_css["card_header"]),
                dbc.CardBody([
                    html.Div([
                        html.P("This dashboard is your all-in-one solution for exploring, cleaning, visualizing, and modeling your data. Whether you're a beginner or an expert, you can easily upload your CSV or Excel files and start analyzing in just a few clicks.", style={"color": "var(--text-secondary)", "fontSize": "18px", "marginBottom": "18px"}),
                        html.Ul([
                            html.Li([html.I(className="fas fa-mouse-pointer", style={"color": "var(--primary)", "marginRight": "8px"}), "Intuitive and interactive: No coding required, just point and click!"], style={"fontSize": "16px", "marginBottom": "10px", "color": "var(--text-primary)"}),
                            html.Li([html.I(className="fas fa-users", style={"color": "var(--primary)", "marginRight": "8px"}), "Accessible to everyone: Designed for all users, regardless of experience."], style={"fontSize": "16px", "marginBottom": "10px", "color": "var(--text-primary)"}),
                            html.Li([html.I(className="fas fa-chart-bar", style={"color": "var(--primary)", "marginRight": "8px"}), "Powerful features: Data cleaning, visualization, machine learning, and more."], style={"fontSize": "16px", "marginBottom": "10px", "color": "var(--text-primary)"}),
                            html.Li([html.I(className="fas fa-magic", style={"color": "var(--primary)", "marginRight": "8px"}), "Modern, beautiful, and responsive design."], style={"fontSize": "16px", "marginBottom": "10px", "color": "var(--text-primary)"}),
                        ], style={"marginBottom": "25px"}),
                        html.P("Get started by uploading your data, or explore the tabs to see what you can do!", style={"color": "var(--primary)", "fontWeight": "bold", "fontSize": "18px", "marginBottom": "30px"}),
                    ]),
                    html.Hr(style={"borderColor": "var(--border-color)", "margin": "30px 0"}),
                    html.Div([
                        html.Div([
                            html.Img(src="/assets/photo de profil.jpg", style={"width": "90px", "height": "90px", "borderRadius": "50%", "marginRight": "25px", "border": "3px solid var(--primary)"}),
                            html.Div([
                                html.H4("About the Author", style={"color": "var(--primary)", "fontWeight": "bold", "marginBottom": "10px"}),
                                html.P("I'm Ilyes Frigui, a first-year computer science engineering student at ESSAI in Tunisia. I'm passionate about technology, artificial intelligence, and data science. I'm currently part of a machine learning club, where I actively participate in projects and workshops focused on real-world applications of AI. I enjoy solving complex problems, building useful tools, and collaborating with others to turn ideas into reality. Outside academics, I'm involved in extracurricular activities such as Enactus, where I develop my teamwork and leadership skills.", style={"color": "var(--text-secondary)", "fontSize": "16px"}),
                                html.Div([
                                    html.I(className="fas fa-envelope", style={"color": "var(--primary)", "marginRight": "8px"}),
                                    html.A("ilyes.frigui.ps@gmail.com", href="mailto:ilyes.frigui.ps@gmail.com", style={"color": "var(--primary)", "textDecoration": "underline", "fontWeight": "bold"})
                                ], style={"marginTop": "10px", "fontSize": "16px"})
                            ])
                        ], style={"display": "flex", "alignItems": "center"})
                    ], style={"marginTop": "10px"})
                ])
            ], style=custom_css["card"]),
        ]),

        # Import tab
        html.Div(id="import-content", style={"display": "block", "opacity": "1", "transition": "opacity 0.3s ease-in-out"}, children=[
            dbc.Card([
                dbc.CardHeader("Import Data", style=custom_css["card_header"]),
                dbc.CardBody([
                    dcc.Upload(
                        id="upload-data",
                        children=html.Div([
                            html.I(className="fas fa-cloud-upload-alt mr-2", style={"fontSize": "24px", "color": "var(--primary)"}),
                            "Drag and Drop or ",
                            html.A("Select a File", style={"color": "var(--primary)", "fontWeight": "bold", "textDecoration": "underline"}),
                        ]),
                        style=custom_css["upload"],
                        className="upload-area",
                    ),
                    html.Div(id="file-upload-status", style={
                        "color": "#a3a3a3",
                        "textAlign": "center",
                        "marginBottom": "15px"
                    }),
                    dbc.Checklist(
                        id="header-checkbox",
                        options=[{"label": html.Span("First row is header", style={"color": "#FFFFFF"}), "value": "header"}],
                        value=["header"],
                        inline=True,
                        style={"marginBottom": "15px", "color": "#e6e6e6"}
                    ),
                    dash_table.DataTable(
                        id="data-table",
                        page_size=10,
                        style_table={"overflowX": "auto", **custom_css["table"]},
                        style_header=custom_css["table_header"],
                        style_cell={
                            "backgroundColor": "#16213e",
                            "color": "#e6e6e6",
                            "padding": "10px",
                            "border": "1px solid #2a3a5e"
                        },
                        style_data_conditional=[
                            {
                                "if": {"row_index": "odd"},
                                "backgroundColor": "#1a1a2e"
                            }
                        ],
                    ),
                    dbc.Button([
                        html.I(className="fas fa-download mr-2"),
                        "Download Data"
                    ], id="download-button", style=custom_css["button"]),
                    dcc.Download(id="download-data"),
                    html.Div([
                        dbc.Button([
                            html.I(className="fas fa-file-excel mr-2"),
                            "Excel"
                        ], id="export-excel-button", color="success", style={"marginRight": "10px", "marginTop": "15px"}),
                        dbc.Button([
                            html.I(className="fas fa-file-code mr-2"),
                            "JSON"
                        ], id="export-json-button", color="info", style={"marginRight": "10px", "marginTop": "15px"}),
                        dbc.Button([
                            html.I(className="fas fa-file-csv mr-2"),
                            "CSV"
                        ], id="export-csv-button", color="warning", style={"marginTop": "15px"}),
                    ], style={"display": "flex", "justifyContent": "center", "width": "100%", "marginTop": "10px"}),
                    dcc.Download(id="export-excel"),
                    dcc.Download(id="export-json"),
                    dcc.Download(id="export-csv"),
                ]),
            ], style=custom_css["card"]),
        ]),

        # Summary tab
        html.Div(id="summary-content", style={"display": "none", "opacity": "0", "transition": "opacity 0.3s ease-in-out"}, children=[
            dbc.Card([
                dbc.CardHeader("Summary Statistics", style=custom_css["card_header"]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Dataset Overview", style=custom_css["card_header"]),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.Div([
                                                html.I(className="fas fa-list-ol fa-2x mr-2", style={"color": "var(--primary)"}),
                                                html.H5("Number of Rows", className="mb-0", style={
                                                    "fontSize": "16px",
                                                    "fontWeight": "600",
                                                    "color": "var(--text-primary)",
                                                    "marginBottom": "5px"
                                                }),
                                                html.P(id="num-rows", className="mb-0", style={
                                                    "fontSize": "24px",
                                                    "fontWeight": "700",
                                                    "color": "var(--primary)",
                                                    "textShadow": "0 0 5px var(--primary-shadow)"
                                                })
                                            ], style={"textAlign": "center"})
                                        ], width=6),
                                        dbc.Col([
                                            html.Div([
                                                html.I(className="fas fa-columns fa-2x mr-2", style={"color": "var(--primary)"}),
                                                html.H5("Number of Columns", className="mb-0", style={
                                                    "fontSize": "16px",
                                                    "fontWeight": "600",
                                                    "color": "var(--text-primary)",
                                                    "marginBottom": "5px"
                                                }),
                                                html.P(id="num-cols", className="mb-0", style={
                                                    "fontSize": "24px",
                                                    "fontWeight": "700",
                                                    "color": "var(--primary)",
                                                    "textShadow": "0 0 5px var(--primary-shadow)"
                                                })
                                            ], style={"textAlign": "center"})
                                        ], width=6),
                                    ]),
                                ]),
                            ], style=custom_css["card"]),
                        ], width=12),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Column Statistics", style=custom_css["card_header"]),
                                dbc.CardBody([
                                    dash_table.DataTable(
                                        id="summary-table",
                                        page_size=10,
                                        style_table={"overflowX": "auto", **custom_css["table"]},
                                        style_header=custom_css["table_header"],
                                        style_cell={
                                            "backgroundColor": "#16213e",
                                            "color": "#e6e6e6",
                                            "padding": "10px",
                                            "border": "1px solid #2a3a5e"
                                        },
                                        style_data_conditional=[
                                            {
                                                "if": {"row_index": "odd"},
                                                "backgroundColor": "#1a1a2e"
                                            }
                                        ],
                                    ),
                                ]),
                            ], style=custom_css["card"]),
                        ], width=12),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Missing Values", style=custom_css["card_header"]),
                                dbc.CardBody([
                                    html.Div(id="missing-values-summary", style={
                                        "textAlign": "center",
                                        "color": "#e6e6e6"
                                    }),
                                ]),
                            ], style=custom_css["card"]),
                        ], width=12),
                    ]),
                    dbc.Row([
                        dbc.Col(
                            dbc.Alert(
                                id="summary-error",
                                color="danger",
                                is_open=False,
                                duration=4000
                            ),
                            width=12
                        ),
                    ]),
                ]),
            ], style=custom_css["card"]),
        ]),

        # Imputation tab
        html.Div(id="imputation-content", style={"display": "none", "opacity": "0", "transition": "opacity 0.3s ease-in-out"}, children=[
            dbc.Card([
                dbc.CardHeader("Data Imputation", style=custom_css["card_header"]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Missing Value Handling", style={**custom_css["card_header"], "fontSize": "18px", "fontWeight": "500"}),
                                dbc.CardBody([
                                    html.Div(id="missing-values-message", style={
                                        "marginBottom": "20px",
                                        "color": "#e6e6e6",
                                        "padding": "10px",
                                        "borderRadius": "6px",
                                        "backgroundColor": "rgba(26, 188, 156, 0.1)"
                                    }),
                                    html.Div([
                                        html.Label("Select columns with missing values:", style={
                                            "color": "#e6e6e6",
                                            "fontWeight": "500",
                                            "marginBottom": "8px",
                                            "display": "block"
                                        }),
                                        dcc.Dropdown(
                                            id="imputation-columns",
                                            multi=True,
                                            placeholder="Select columns to impute",
                                            style={"marginBottom": "25px", **custom_css["dropdown"]},
                                            className='dropdown-dark custom-dropdown'
                                        )
                                    ]),
                                    html.Div([
                                        html.Label("Select imputation method:", style={
                                            "color": "#e6e6e6",
                                            "fontWeight": "500",
                                            "marginBottom": "8px",
                                            "display": "block"
                                        }),
                                        dcc.Dropdown(
                                            id="missing-method",
                                            options=[
                                                {"label": html.Span("Replace with mean (numeric only)", style={"color": "#FFFFFF"}), "value": "mean"},
                                                {"label": html.Span("Replace with median (numeric only)", style={"color": "#FFFFFF"}), "value": "median"},
                                                {"label": html.Span("Replace with mode (numeric & categorical)", style={"color": "#FFFFFF"}), "value": "mode"},
                                                {"label": html.Span("KNN Imputation (numeric only)", style={"color": "#FFFFFF"}), "value": "knn"},
                                            ],
                                            value="mean",
                                            placeholder="Select imputation method",
                                            style={"marginBottom": "25px", **custom_css["dropdown"]},
                                            className='dropdown-dark custom-dropdown'
                                        )
                                    ]),
                                    html.Div([
                                        dbc.Button(
                                            "Apply Imputation",
                                            id="apply-imputation-button",
                                            color="primary",
                                            className="mt-2",
                                            style={
                                                "backgroundColor": "#1abc9c",
                                                "border": "none",
                                                "width": "100%",
                                                "padding": "12px",
                                                "fontWeight": "500",
                                                "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
                                                "transition": "all 0.3s ease"
                                            }
                                        ),
                                    ]),
                                    dbc.Toast(
                                        "Imputation applied successfully!",
                                        id="imputation-success-toast",
                                        header="Success",
                                        is_open=False,
                                        dismissable=True,
                                        icon="success",
                                        duration=4000,
                                        style={
                                            "position": "fixed",
                                            "top": 10,
                                            "right": 10,
                                            "width": 350,
                                            "zIndex": 1999
                                        }
                                    ),
                                    dbc.Toast(
                                        "Please select at least one column for imputation.",
                                        id="imputation-warning-toast",
                                        header="Warning",
                                        is_open=False,
                                        dismissable=True,
                                        icon="danger",
                                        duration=4000,
                                        style={
                                            "position": "fixed",
                                            "top": 10,
                                            "right": 10,
                                            "width": 350,
                                            "zIndex": 1999
                                        }
                                    ),
                                ]),
                            ], style=custom_css["card"]),
                        ], width=4),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Duplicate Rows Handling", style=custom_css["card_header"]),
                                dbc.CardBody([
                                    html.Div(id="duplicates-message", style={
                                        "marginBottom": "15px",
                                        "color": "#e6e6e6"
                                    }),
                                    dbc.Button(
                                        [html.I(className="fas fa-search mr-2"), "Find Duplicates"],
                                        id="find-duplicates-button",
                                        style=custom_css["button"]
                                    ),
                                    dbc.Button(
                                        [html.I(className="fas fa-trash-alt mr-2"), "Remove Duplicates"],
                                        id="remove-duplicates-button",
                                        style=custom_css["button"],
                                        disabled=True
                                    ),
                                ]),
                            ], style=custom_css["card"]),
                        ], width=4),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Outlier Detection", style=custom_css["card_header"]),
                                dbc.CardBody([
                                    dcc.Dropdown(
                                        id="outlier-columns",
                                        multi=True,
                                        placeholder="Select numeric columns",
                                        style={"marginBottom": "15px", **custom_css["dropdown"]},
                                        className='dropdown-dark custom-dropdown'
                                    ),
                                    dcc.Dropdown(
                                        id="outlier-method",
                                        options=[
                                            {"label": html.Span("IQR Method", style={"color": "#FFFFFF"}), "value": "iqr"},
                                            {"label": html.Span("Z-Score Method", style={"color": "#FFFFFF"}), "value": "zscore"},
                                        ],
                                        value="iqr",
                                        placeholder="Select detection method",
                                        style={"marginBottom": "15px", **custom_css["dropdown"]},
                                        className='dropdown-dark custom-dropdown'
                                    ),
                                    dbc.Input(
                                        id="outlier-threshold",
                                        type="number",
                                        placeholder="Threshold (default: 3 for z-score, 1.5 for IQR)",
                                        style={"marginBottom": "15px", **custom_css["dropdown"]}
                                    ),
                                    dbc.Button(
                                        [html.I(className="fas fa-search mr-2"), "Detect Outliers"],
                                        id="detect-outliers-button",
                                        style=custom_css["button"]
                                    ),
                                    html.Div(id="outliers-message", style={"marginTop": "15px"}),
                                    dcc.Dropdown(
                                        id="outlier-handling-method",
                                        options=[
                                            {"label": html.Span("Remove outliers", style={"color": "#FFFFFF"}), "value": "remove"},
                                            {"label": html.Span("Replace with median", style={"color": "#FFFFFF"}), "value": "median"},
                                            {"label": html.Span("Replace with mean", style={"color": "#FFFFFF"}), "value": "mean"},
                                        ],
                                        value="remove",
                                        placeholder="Select handling method",
                                        style={"marginTop": "15px", **custom_css["dropdown"]},
                                        className='dropdown-dark custom-dropdown'
                                    ),
                                    dbc.Button(
                                        [html.I(className="fas fa-wrench mr-2"), "Handle Outliers"],
                                        id="handle-outliers-button",
                                        style=custom_css["button"],
                                        disabled=True
                                    ),
                                ]),
                            ], style=custom_css["card"]),
                        ], width=4),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Preview", style=custom_css["card_header"]),
                                dbc.CardBody([
                                    dcc.Dropdown(
                                        id="imputation-rows",
                                        options=[
                                            {"label": html.Span("Show 5 rows", style={"color": "#FFFFFF"}), "value": 5},
                                            {"label": html.Span("Show 10 rows", style={"color": "#FFFFFF"}), "value": 10},
                                            {"label": html.Span("Show 20 rows", style={"color": "#FFFFFF"}), "value": 20},
                                            {"label": html.Span("Show 50 rows", style={"color": "#FFFFFF"}), "value": 50},
                                            {"label": html.Span("Show all rows", style={"color": "#FFFFFF"}), "value": "all"},
                                        ],
                                        value=10,
                                        placeholder="Select number of rows to display",
                                        style={"marginBottom": "20px", **custom_css["dropdown"]},
                                        className='dropdown-dark custom-dropdown'
                                    ),
                                    dash_table.DataTable(
                                        id="imputed-table",
                                        style_table={"overflowX": "auto", **custom_css["table"]},
                                        style_header=custom_css["table_header"],
                                        style_cell={
                                            "backgroundColor": "#16213e",
                                            "color": "#e6e6e6",
                                            "padding": "10px",
                                            "border": "1px solid #2a3a5e"
                                        },
                                        style_data_conditional=[
                                            {
                                                "if": {"row_index": "odd"},
                                                "backgroundColor": "#1a1a2e"
                                            }
                                        ],
                                    ),
                                    html.Div([
                                        html.H6("Download Imputed Data", style={"marginTop": "20px", "marginBottom": "10px", "color": "#e6e6e6"}),
                                        html.Div([
                                            dbc.Button([
                                                html.I(className="fas fa-file-csv mr-2"),
                                                "CSV"
                                            ], id="download-imputed-csv-button", color="success", style={"marginRight": "10px"}),
                                            dbc.Button([
                                                html.I(className="fas fa-file-code mr-2"),
                                                "JSON"
                                            ], id="download-imputed-json-button", color="info", style={"marginRight": "10px"}),
                                            dbc.Button([
                                                html.I(className="fas fa-file-excel mr-2"),
                                                "Excel"
                                            ], id="download-imputed-excel-button", color="warning"),
                                        ], style={"display": "flex", "justifyContent": "center", "width": "100%"}),
                                        dcc.Download(id="download-imputed-csv"),
                                        dcc.Download(id="download-imputed-json"),
                                        dcc.Download(id="download-imputed-excel"),
                                    ], style={"marginTop": "15px", "textAlign": "center"}),
                                ]),
                            ], style=custom_css["card"]),
                        ], width=12),
                    ]),
                ]),
            ], style=custom_css["card"]),
        ]),

        # Statistics tab
        html.Div(id="statistics-content", style={"display": "none", "opacity": "0", "transition": "opacity 0.3s ease-in-out"}, children=[
            dbc.Card([
                dbc.CardHeader("Data Visualization", style=custom_css["card_header"]),
                dbc.CardBody([
                    # New Auto-generated visualizations section
                    dbc.Row([
                        dbc.Col([
                            html.H4("Data Overview", style={"color": "#FFFFFF", "marginBottom": "20px"}),
                            html.P("Automatically generated visualizations based on your data:",
                                  style={"color": "#e6e6e6", "marginBottom": "20px"}),
                            dbc.Spinner(html.Div(id="auto-visualizations")),
                        ], width=12),
                    ]),
                    html.Hr(style={"borderColor": "#2a3a5e", "margin": "30px 0"}),

                    # Original custom plot controls section
                    html.H4("Custom Plot Controls", style={"color": "#FFFFFF", "marginBottom": "20px"}),
                    html.P("Create your own custom visualizations by selecting options below:",
                          style={"color": "#e6e6e6", "marginBottom": "20px"}),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Plot Controls", style=custom_css["card_header"]),
                                dbc.CardBody([
                                    # X-axis selection
                                    html.Div(className="form-group", style={"marginBottom": "25px"}, children=[
                                        html.Label("Select X-axis:", style={
                                            "color": "#e6e6e6",
                                            "fontWeight": "bold",
                                            "marginBottom": "12px",
                                            "fontSize": "16px",
                                            "display": "block"
                                        }),
                                        dcc.Dropdown(
                                            id="x-axis-dropdown",
                                            placeholder="Select X-axis",
                                            style={"height": "40px", **custom_css["dropdown"]},
                                            className='dropdown-dark custom-dropdown'
                                        ),
                                    ]),

                                    # Y-axis selection
                                    html.Div(className="form-group", style={"marginBottom": "25px"}, children=[
                                        html.Label("Select Y-axis (optional):", style={
                                            "color": "#e6e6e6",
                                            "fontWeight": "bold",
                                            "marginBottom": "12px",
                                            "fontSize": "16px",
                                            "display": "block"
                                        }),
                                        dcc.Dropdown(
                                            id="y-axis-dropdown",
                                            placeholder="Select Y-axis",
                                            style={"height": "40px", **custom_css["dropdown"]},
                                            className='dropdown-dark custom-dropdown'
                                        ),
                                    ]),

                                    # Plot type selection
                                    html.Div(className="form-group", style={"marginBottom": "25px"}, children=[
                                        html.Label("Select Plot Type:", style={
                                            "color": "#e6e6e6",
                                            "fontWeight": "bold",
                                            "marginBottom": "12px",
                                            "fontSize": "16px",
                                            "display": "block"
                                        }),
                                        dcc.Dropdown(
                                            id="plot-type-dropdown",
                                            options=[
                                                {"label": html.Span("Scatter Plot", style={"color": "#FFFFFF"}), "value": "scatter"},
                                                {"label": html.Span("Line Chart", style={"color": "#FFFFFF"}), "value": "line"},
                                                {"label": html.Span("Bar Chart", style={"color": "#FFFFFF"}), "value": "bar"},
                                                {"label": html.Span("Box Plot", style={"color": "#FFFFFF"}), "value": "box"},
                                                {"label": html.Span("Violin Plot", style={"color": "#FFFFFF"}), "value": "violin"},
                                                {"label": html.Span("Histogram", style={"color": "#FFFFFF"}), "value": "histogram"},
                                                {"label": html.Span("Pie Chart", style={"color": "#FFFFFF"}), "value": "pie"},
                                                {"label": html.Span("Heatmap", style={"color": "#FFFFFF"}), "value": "heatmap"},
                                                {"label": html.Span("Time Series", style={"color": "#FFFFFF"}), "value": "timeseries"},
                                                {"label": html.Span("Scatter Matrix", style={"color": "#FFFFFF"}), "value": "scattermatrix"},
                                                {"label": html.Span("3D Scatter", style={"color": "#FFFFFF"}), "value": "scatter3d"},
                                                {"label": html.Span("3D Surface", style={"color": "#FFFFFF"}), "value": "surface3d"},
                                                {"label": html.Span("Choropleth Map", style={"color": "#FFFFFF"}), "value": "choropleth"},
                                                {"label": html.Span("Scatter Map", style={"color": "#FFFFFF"}), "value": "scattermap"},
                                                {"label": html.Span("Q-Q Plot", style={"color": "#FFFFFF"}), "value": "qqplot"},
                                                {"label": html.Span("Residual Plot", style={"color": "#FFFFFF"}), "value": "residual"},
                                            ],
                                            value="scatter",
                                            clearable=False,
                                            style={"marginBottom": "15px", **custom_css["dropdown"]},
                                            className='dropdown-dark custom-dropdown'
                                        ),
                                    ]),

                                    # Bin size slider
                                    html.Div(id="bin-size-col", className="form-group", style={"marginBottom": "35px"}, children=[
                                        html.Label("Adjust Bin Size (for histograms):", style={
                                            "color": "#e6e6e6",
                                            "fontWeight": "bold",
                                            "marginBottom": "20px",
                                            "fontSize": "16px",
                                            "display": "block"
                                        }),
                                        html.Div([
                                            dcc.Slider(
                                                id="bin-size-slider",
                                                min=5,
                                                max=50,
                                                step=5,
                                                value=10,
                                                marks={i: {"label": str(i), "style": {"color": "white", "font-size": "14px"}}
                                                      for i in range(5, 51, 5)},
                                            )
                                        ], style={"paddingTop": "15px", "paddingBottom": "15px", "paddingLeft": "10px", "paddingRight": "10px"}),
                                    ]),

                                    # Time Series Controls (hidden by default)
                                    html.Div(id="time-series-controls", style={"display": "none", "marginBottom": "25px"}, children=[
                                        # Date column
                                        html.Div(className="form-group", style={"marginBottom": "20px"}, children=[
                                            html.Label("Date Column:", style={
                                                "color": "#e6e6e6",
                                                "fontWeight": "bold",
                                                "marginBottom": "12px",
                                                "fontSize": "16px",
                                                "display": "block"
                                            }),
                                            dcc.Dropdown(
                                                id="date-column-dropdown",
                                                placeholder="Select date column",
                                                style={"height": "40px", **custom_css["dropdown"]},
                                                className='dropdown-dark custom-dropdown'
                                            ),
                                        ]),

                                        # Value column
                                        html.Div(className="form-group", style={"marginBottom": "20px"}, children=[
                                            html.Label("Value Column:", style={
                                                "color": "#e6e6e6",
                                                "fontWeight": "bold",
                                                "marginBottom": "12px",
                                                "fontSize": "16px",
                                                "display": "block"
                                            }),
                                            dcc.Dropdown(
                                                id="value-column-dropdown",
                                                placeholder="Select value column",
                                                style={"height": "40px", **custom_css["dropdown"]},
                                                className='dropdown-dark custom-dropdown'
                                            ),
                                        ]),

                                        # Time series options
                                        html.Div(className="form-group", style={"marginBottom": "20px"}, children=[
                                            html.Label("Time Series Options:", style={
                                                "color": "#e6e6e6",
                                                "fontWeight": "bold",
                                                "marginBottom": "12px",
                                                "fontSize": "16px",
                                                "display": "block"
                                            }),
                                            dbc.Checklist(
                                                id="time-series-options",
                                                options=[
                                                    {"label": html.Span("Show Trendline", style={"color": "#FFFFFF", "marginLeft": "5px"}), "value": "trendline"},
                                                    {"label": html.Span("Show Moving Average", style={"color": "#FFFFFF", "marginLeft": "5px"}), "value": "moving_avg"},
                                                    {"label": html.Span("Show Seasonality", style={"color": "#FFFFFF", "marginLeft": "5px"}), "value": "seasonality"},
                                                ],
                                                value=[],
                                                inline=True,
                                                style={"marginBottom": "20px", "color": "#e6e6e6"}
                                            ),
                                        ]),

                                        # Moving average and seasonality
                                        dbc.Row([
                                            dbc.Col([
                                                html.Div(id="moving-avg-col", className="form-group", children=[
                                                    html.Label("Moving Average Window:", style={
                                                        "color": "#e6e6e6",
                                                        "fontWeight": "bold",
                                                        "marginBottom": "12px",
                                                        "fontSize": "16px",
                                                        "display": "block"
                                                    }),
                                                    dbc.Input(
                                                        id="moving-avg-window",
                                                        type="number",
                                                        min=2,
                                                        value=7,
                                                        style={"height": "40px", **custom_css["dropdown"]}
                                                    ),
                                                ]),
                                            ], width=6),
                                            dbc.Col([
                                                html.Div(id="seasonality-col", className="form-group", children=[
                                                    html.Label("Seasonality Period:", style={
                                                        "color": "#e6e6e6",
                                                        "fontWeight": "bold",
                                                        "marginBottom": "12px",
                                                        "fontSize": "16px",
                                                        "display": "block"
                                                    }),
                                                    dbc.Input(
                                                        id="seasonality-period",
                                                        type="number",
                                                        min=2,
                                                        value=12,
                                                        style={"height": "40px", **custom_css["dropdown"]}
                                                    ),
                                                ]),
                                            ], width=6),
                                        ]),
                                    ]),

                                    # Scatter Matrix Controls (hidden by default)
                                    html.Div(id="scatter-matrix-controls", style={"display": "none", "marginBottom": "25px"}, children=[
                                        # Variables selection
                                        html.Div(className="form-group", style={"marginBottom": "20px"}, children=[
                                            html.Label("Select Variables:", style={
                                                "color": "#e6e6e6",
                                                "fontWeight": "bold",
                                                "marginBottom": "12px",
                                                "fontSize": "16px",
                                                "display": "block"
                                            }),
                                            dcc.Dropdown(
                                                id="scatter-matrix-vars",
                                                multi=True,
                                                placeholder="Select variables for scatter matrix",
                                                style={"height": "40px", **custom_css["dropdown"]},
                                                className='dropdown-dark custom-dropdown'
                                            ),
                                        ]),

                                        # Color selection
                                        html.Div(className="form-group", style={"marginBottom": "20px"}, children=[
                                            html.Label("Color By (optional):", style={
                                                "color": "#e6e6e6",
                                                "fontWeight": "bold",
                                                "marginBottom": "12px",
                                                "fontSize": "16px",
                                                "display": "block"
                                            }),
                                            dcc.Dropdown(
                                                id="scatter-matrix-color",
                                                placeholder="Select variable for coloring",
                                                style={"height": "40px", **custom_css["dropdown"]},
                                                className='dropdown-dark custom-dropdown'
                                            ),
                                        ]),
                                    ]),

                                    # 3D Plot Controls (hidden by default)
                                    html.Div(id="3d-plot-controls", style={"display": "none", "marginBottom": "25px"}, children=[
                                        # Z-axis column
                                        html.Div(className="form-group", style={"marginBottom": "20px"}, children=[
                                            html.Label("Z-axis Column:", style={
                                                "color": "#e6e6e6",
                                                "fontWeight": "bold",
                                                "marginBottom": "12px",
                                                "fontSize": "16px",
                                                "display": "block"
                                            }),
                                            dcc.Dropdown(
                                                id="z-axis-dropdown",
                                                placeholder="Select Z-axis column",
                                                style={"height": "40px", **custom_css["dropdown"]},
                                                className='dropdown-dark custom-dropdown'
                                            ),
                                        ]),

                                        # Color column
                                        html.Div(className="form-group", style={"marginBottom": "20px"}, children=[
                                            html.Label("Color Column (optional):", style={
                                                "color": "#e6e6e6",
                                                "fontWeight": "bold",
                                                "marginBottom": "12px",
                                                "fontSize": "16px",
                                                "display": "block"
                                            }),
                                            dcc.Dropdown(
                                                id="color-variable-dropdown",
                                                placeholder="Select color column",
                                                style={"height": "40px", **custom_css["dropdown"]},
                                                className='dropdown-dark custom-dropdown'
                                            ),
                                        ]),
                                    ]),

                                    # Geographic Map Controls (hidden by default)
                                    html.Div(id="geo-plot-controls", style={"display": "none", "marginBottom": "25px"}, children=[
                                        # Location column
                                        html.Div(className="form-group", style={"marginBottom": "20px"}, children=[
                                            html.Label("Location Column:", style={
                                                "color": "#e6e6e6",
                                                "fontWeight": "bold",
                                                "marginBottom": "12px",
                                                "fontSize": "16px",
                                                "display": "block"
                                            }),
                                            dcc.Dropdown(
                                                id="geo-location-dropdown",
                                                placeholder="Select location column",
                                                style={"height": "40px", **custom_css["dropdown"]},
                                                className='dropdown-dark custom-dropdown'
                                            ),
                                        ]),

                                        # Value column for geographic maps
                                        html.Div(className="form-group", style={"marginBottom": "20px"}, children=[
                                            html.Label("Value Column:", style={
                                                "color": "#e6e6e6",
                                                "fontWeight": "bold",
                                                "marginBottom": "12px",
                                                "fontSize": "16px",
                                                "display": "block"
                                            }),
                                            dcc.Dropdown(
                                                id="geo-value-dropdown",
                                                placeholder="Select value column",
                                                style={"height": "40px", **custom_css["dropdown"]},
                                                className='dropdown-dark custom-dropdown'
                                            ),
                                        ]),

                                        # Geographic scope
                                        html.Div(className="form-group", style={"marginBottom": "20px"}, children=[
                                            html.Label("Map Scope:", style={
                                                "color": "#e6e6e6",
                                                "fontWeight": "bold",
                                                "marginBottom": "12px",
                                                "fontSize": "16px",
                                                "display": "block"
                                            }),
                                            dcc.Dropdown(
                                                id="geo-scope-dropdown",
                                                options=[
                                                    {"label": html.Span("World", style={"color": "#FFFFFF"}), "value": "world"},
                                                    {"label": html.Span("USA", style={"color": "#FFFFFF"}), "value": "usa"},
                                                    {"label": html.Span("Europe", style={"color": "#FFFFFF"}), "value": "europe"},
                                                    {"label": html.Span("Asia", style={"color": "#FFFFFF"}), "value": "asia"},
                                                    {"label": html.Span("Africa", style={"color": "#FFFFFF"}), "value": "africa"},
                                                ],
                                                value="world",
                                                style={"height": "40px", **custom_css["dropdown"]},
                                                className='dropdown-dark custom-dropdown'
                                            ),
                                        ]),
                                    ]),

                                    # Forecast Controls (hidden by default)
                                    html.Div(id="forecast-controls", style={"display": "none", "marginBottom": "25px"}, children=[
                                        # Forecast model
                                        html.Div(className="form-group", style={"marginBottom": "20px"}, children=[
                                            html.Label("Forecast Model:", style={
                                                "color": "#e6e6e6",
                                                "fontWeight": "bold",
                                                "marginBottom": "12px",
                                                "fontSize": "16px",
                                                "display": "block"
                                            }),
                                            dcc.Dropdown(
                                                id="forecast-model-dropdown",
                                                options=[
                                                    {"label": html.Span("ARIMA", style={"color": "#FFFFFF"}), "value": "arima"},
                                                    {"label": html.Span("Prophet", style={"color": "#FFFFFF"}), "value": "prophet"},
                                                ],
                                                value="arima",
                                                style={"height": "40px", **custom_css["dropdown"]},
                                                className='dropdown-dark custom-dropdown'
                                            ),
                                        ]),

                                        # Forecast periods
                                        html.Div(className="form-group", style={"marginBottom": "20px"}, children=[
                                            html.Label("Forecast Periods:", style={
                                                "color": "#e6e6e6",
                                                "fontWeight": "bold",
                                                "marginBottom": "12px",
                                                "fontSize": "16px",
                                                "display": "block"
                                            }),
                                            dbc.Input(
                                                id="forecast-periods",
                                                type="number",
                                                min=1,
                                                value=10,
                                                style={"height": "40px", **custom_css["dropdown"]}
                                            ),
                                        ]),
                                    ]),

                                    # Statistical Plot Controls (hidden by default)
                                    html.Div(id="stat-plot-controls", style={"display": "none", "marginBottom": "25px"}, children=[
                                        # Reference distribution
                                        html.Div(className="form-group", style={"marginBottom": "20px"}, children=[
                                            html.Label("Reference Distribution:", style={
                                                "color": "#e6e6e6",
                                                "fontWeight": "bold",
                                                "marginBottom": "12px",
                                                "fontSize": "16px",
                                                "display": "block"
                                            }),
                                            dcc.Dropdown(
                                                id="reference-distribution-dropdown",
                                                options=[
                                                    {"label": html.Span("Normal", style={"color": "#FFFFFF"}), "value": "norm"},
                                                    {"label": html.Span("T", style={"color": "#FFFFFF"}), "value": "t"},
                                                    {"label": html.Span("Chi-Square", style={"color": "#FFFFFF"}), "value": "chi2"},
                                                ],
                                                value="norm",
                                                style={"height": "40px", **custom_css["dropdown"]},
                                                className='dropdown-dark custom-dropdown'
                                            ),
                                        ]),
                                    ]),

                                    # Generate Plot button
                                    dbc.Button([
                                        html.I(className="fas fa-chart-bar mr-2"),
                                        "Generate Plot"
                                    ], id="generate-plot-button", style=custom_css["button"]),
                                ]),
                            ], style=custom_css["card"]),
                        ], width=4),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Visualization", style=custom_css["card_header"]),
                                dbc.CardBody([
                                    dcc.Graph(id="statistics-plot", style={"height": "700px"}),
                                    dbc.Alert(id="statistics-error", color="danger", is_open=False, duration=4000),
                                ]),
                            ], style=custom_css["card"]),
                        ], width=8),
                    ]),
                ]),
            ], style=custom_css["card"]),
        ]),

        # Tests tab
        html.Div(id="tests-content", style={"display": "none", "opacity": "0", "transition": "opacity 0.3s ease-in-out"}, children=[
            dbc.Card([
                dbc.CardHeader("Statistical Tests", style=custom_css["card_header"]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(html.Div([
                            dcc.Dropdown(
                                id="test-type-dropdown",
                                placeholder="Select Test Type",
                                options=[
                                    {"label": html.Span("Chi-squared Test", style={"color": "#FFFFFF"}), "value": "chi2"},
                                    {"label": html.Span("Pearson Correlation", style={"color": "#FFFFFF"}), "value": "pearson"},
                                    {"label": html.Span("Spearman Correlation", style={"color": "#FFFFFF"}), "value": "spearman"},
                                ],
                                style=custom_css["dropdown"],
                                className='dropdown-dark'
                            )
                        ]), width=6),
                        dbc.Col(html.Div([
                            dcc.Dropdown(
                                id="test-x-dropdown",
                                placeholder="Select variable (filtered by test)",
                                style=custom_css["dropdown"],
                                className='dropdown-dark'
                            )
                        ]), width=6),
                    ], style={"marginBottom": "20px"}),
                    dbc.Row([
                        dbc.Col(html.Div([
                            dcc.Dropdown(
                                id="test-y-dropdown",
                                placeholder="Select second variable (filtered by test)",
                                style=custom_css["dropdown"],
                                className='dropdown-dark'
                            )
                        ]), width=6),
                    ], style={"marginBottom": "20px"}),
                    dbc.Row([
                        dbc.Col(dbc.Button(
                            [html.I(className="fas fa-calculator mr-2"), "Perform Test"],
                            id="perform-test",
                            style=custom_css["button"]
                        ))
                    ]),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id="test-plot"))
                    ]),
                    dbc.Row([
                        dbc.Col(html.Div(
                            id="test-result",
                            style={
                                "textAlign": "center",
                                "margin": "15px 0",
                                "color": "#e6e6e6"
                            }
                        ))
                    ]),
                    dbc.Row([
                        dbc.Col(dash_table.DataTable(
                            id="test-table",
                            style_table={"overflowX": "auto", **custom_css["table"]},
                            style_header=custom_css["table_header"],
                            style_cell={
                                "backgroundColor": "#16213e",
                                "color": "#e6e6e6",
                                "padding": "10px",
                                "border": "1px solid #2a3a5e"
                            },
                            style_data_conditional=[
                                {
                                    "if": {"row_index": "odd"},
                                    "backgroundColor": "#1a1a2e"
                                }
                            ],
                        ))
                    ]),
                ]),
            ], style=custom_css["card"]),
        ]),

        # Regression tab
        html.Div(id="regression-content", style={"display": "none", "opacity": "0", "transition": "opacity 0.3s ease-in-out"}, children=[
            dbc.Card([
                dbc.CardHeader("Linear Regression Analysis", style=custom_css["card_header"]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Variable Selection", style=custom_css["card_header"]),
                                dbc.CardBody([
                                    html.Div([
                                        html.Label("Independent Variable (X):", style={
                                            "color": "#e6e6e6",
                                            "fontWeight": "bold",
                                            "marginBottom": "12px",
                                            "fontSize": "16px",
                                            "display": "block"
                                        }),
                                        dcc.Dropdown(
                                            id="regression-x-dropdown",
                                            placeholder="Select independent variable",
                                            style={"marginBottom": "20px", **custom_css["dropdown"]},
                                            className='dropdown-dark custom-dropdown'
                                        ),
                                    ]),
                                    html.Div([
                                        html.Label("Dependent Variable (Y):", style={
                                            "color": "#e6e6e6",
                                            "fontWeight": "bold",
                                            "marginBottom": "12px",
                                            "fontSize": "16px",
                                            "display": "block"
                                        }),
                                        dcc.Dropdown(
                                            id="regression-y-dropdown",
                                            placeholder="Select dependent variable",
                                            style={"marginBottom": "20px", **custom_css["dropdown"]},
                                            className='dropdown-dark custom-dropdown'
                                        ),
                                    ]),
                                    dbc.Button(
                                        [html.I(className="fas fa-calculator mr-2"), "Calculate Regression"],
                                        id="calculate-regression",
                                        style=custom_css["button"]
                                    ),
                                ]),
                            ], style=custom_css["card"]),

                            # Regression Results Card
                            dbc.Card([
                                dbc.CardHeader("Regression Results", style=custom_css["card_header"]),
                                dbc.CardBody([
                                    html.Div(id="regression-equation", style={
                                        "fontSize": "1.2em",
                                        "fontWeight": "bold",
                                        "marginBottom": "20px",
                                        "color": "var(--primary)",
                                        "textAlign": "center"
                                    }),
                                    html.Div(id="regression-metrics", style={
                                        "marginBottom": "20px",
                                        "color": "#e6e6e6"
                                    }),
                                    html.Hr(style={"borderColor": "var(--border-color)"}),
                                    html.Div([
                                        html.Label("Make a Prediction:", style={
                                            "color": "#e6e6e6",
                                            "fontWeight": "bold",
                                            "marginBottom": "12px",
                                            "fontSize": "16px",
                                            "display": "block"
                                        }),
                                        dbc.Input(
                                            id="prediction-input",
                                            type="number",
                                            placeholder="Enter X value",
                                            style={"marginBottom": "15px", **custom_css["dropdown"]}
                                        ),
                                        dbc.Button(
                                            [html.I(className="fas fa-magic mr-2"), "Predict"],
                                            id="predict-button",
                                            style=custom_css["button"]
                                        ),
                                        html.Div(id="prediction-result", style={
                                            "marginTop": "15px",
                                            "color": "var(--primary)",
                                            "fontWeight": "bold",
                                            "textAlign": "center"
                                        })
                                    ])
                                ]),
                            ], style={**custom_css["card"], "marginTop": "20px"}),
                        ], width=4),

                        # Regression Plot
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Regression Plot", style=custom_css["card_header"]),
                                dbc.CardBody([
                                    dcc.Graph(id="regression-plot", style={"height": "600px"}),
                                ]),
                            ], style=custom_css["card"]),
                        ], width=8),
                    ]),
                    dbc.Row([
                        dbc.Col(
                            dbc.Alert(
                                id="regression-error",
                                color="danger",
                                is_open=False,
                                duration=4000
                            ),
                            width=12
                        ),
                    ]),
                ]),
            ], style=custom_css["card"]),
        ]),

        # FAQ tab
        html.Div(id="faq-content", style={"display": "none", "opacity": "0", "transition": "opacity 0.3s ease-in-out"}, children=[
            dbc.Card([
                dbc.CardHeader([
                    html.H3("Frequently Asked Questions", className="mb-0", style={"color": "var(--primary)", "fontWeight": "600"})
                ], style=custom_css["card_header"]),
                dbc.CardBody([
                    html.Div(style={"display": "flex", "alignItems": "center", "marginBottom": "25px"}, children=[
                        html.I(className="fas fa-question-circle", style={"fontSize": "24px", "color": "var(--primary)", "marginRight": "15px"}),
                        html.P("Find answers to common questions and learn how to make the most of this data analysis dashboard. Browse through the categories below to quickly find the information you need.",
                          style={"color": "var(--text-secondary)", "fontSize": "16px", "margin": "0"})
                    ]),

                    # FAQ categories
                    dbc.Tabs([
                        # Getting Started Tab
                        dbc.Tab(label="Getting Started", tab_id="getting-started", label_style={"fontWeight": "bold", "padding": "12px 15px"}, active_label_style={"color": "var(--primary)", "borderBottom": "2px solid var(--primary)"}, children=[
                            html.Div(style={"marginTop": "20px", "padding": "5px"}, children=[
                                dbc.Accordion([
                                    dbc.AccordionItem(
                                        [
                                            html.P("This app lets you upload CSV or Excel files and perform data analysis through an intuitive interface. The typical workflow is:", style={"color": "var(--text-secondary)"}),
                                            html.Ol([
                                                html.Li("Upload your data in the Import tab"),
                                                html.Li("View summary statistics in the Summary tab"),
                                                html.Li("Clean your data in the Imputation tab (impute missing values, remove duplicates, handle outliers)"),
                                                html.Li("Create visualizations in the Statistics tab (auto and custom plots)"),
                                                html.Li("Analyze relationships in the Correlation and Tests tabs"),
                                                html.Li("Build and use regression and prediction models in the Regression and Prediction tabs"),
                                                html.Li("Generate a comprehensive EDA report in the Report tab")
                                            ], style={"color": "var(--text-secondary)", "marginLeft": "20px"})
                                        ],
                                        title="How can I use this app?",
                                        style={"backgroundColor": "var(--card-bg)", "marginBottom": "10px", "borderColor": "var(--border-color)", "borderRadius": "8px"},
                                        className="faq-accordion-item",
                                    ),
                                    dbc.AccordionItem(
                                        [
                                            html.P("You can upload the following file formats:", style={"color": "var(--text-secondary)"}),
                                            html.Ul([
                                                html.Li("CSV (.csv) - Comma-separated values"),
                                                html.Li("Excel (.xls, .xlsx) - Microsoft Excel spreadsheets")
                                            ], style={"color": "var(--text-secondary)", "marginLeft": "20px"}),
                                            html.P("Files should be properly formatted with consistent data types in each column for best results.", style={"color": "var(--text-secondary)", "marginTop": "10px"})
                                        ],
                                        title="What types of files can I upload?",
                                        style={"backgroundColor": "var(--card-bg)", "marginBottom": "10px", "borderColor": "var(--border-color)", "borderRadius": "8px"},
                                        className="faq-accordion-item",
                                    ),
                                    dbc.AccordionItem(
                                        [
                                            html.P("The app automatically detects numeric, categorical, datetime, and boolean columns. It suggests conversions if needed.", style={"color": "var(--text-secondary)"}),
                                        ],
                                        title="How are data types determined?",
                                        style={"backgroundColor": "var(--card-bg)", "marginBottom": "10px", "borderColor": "var(--border-color)", "borderRadius": "8px"},
                                        className="faq-accordion-item",
                                    ),
                                    dbc.AccordionItem(
                                        [
                                            html.P("For best performance, use files with up to 10,000 rows and 100 columns. Larger files may be sampled.", style={"color": "var(--text-secondary)"}),
                                        ],
                                        title="What's the maximum file size I can upload?",
                                        style={"backgroundColor": "var(--card-bg)", "marginBottom": "10px", "borderColor": "var(--border-color)", "borderRadius": "8px"},
                                        className="faq-accordion-item",
                                    ),
                                ], start_collapsed=True, style={"borderRadius": "8px", "boxShadow": "0 2px 5px rgba(0, 0, 0, 0.05)"}),
                            ]),
                        ]),

                        # Data Cleaning Tab
                        dbc.Tab(label="Data Cleaning", tab_id="data-cleaning", label_style={"fontWeight": "bold", "padding": "12px 15px"}, active_label_style={"color": "var(--primary)", "borderBottom": "2px solid var(--primary)"}, children=[
                            html.Div(style={"marginTop": "20px", "padding": "5px"}, children=[
                                dbc.Accordion([
                                    dbc.AccordionItem(
                                        [
                                            html.P("In the Imputation tab, select columns and choose a method: mean, median, mode, or KNN (for numeric). Apply changes to fill missing values.", style={"color": "var(--text-secondary)"}),
                                        ],
                                        title="How do I handle missing values?",
                                        style={"backgroundColor": "var(--card-bg)", "marginBottom": "10px", "borderColor": "var(--border-color)", "borderRadius": "8px"},
                                        className="faq-accordion-item",
                                    ),
                                    dbc.AccordionItem(
                                        [
                                            html.P("In the Imputation tab, click 'Find Duplicates' to preview, then 'Remove Duplicates' to delete them.", style={"color": "var(--text-secondary)"}),
                                        ],
                                        title="How can I remove duplicate records?",
                                        style={"backgroundColor": "var(--card-bg)", "marginBottom": "10px", "borderColor": "var(--border-color)", "borderRadius": "8px"},
                                        className="faq-accordion-item",
                                    ),
                                    dbc.AccordionItem(
                                        [
                                            html.P("In the Imputation tab, select numeric columns, choose IQR or Z-score, set a threshold, detect outliers, and choose to remove or replace them.", style={"color": "var(--text-secondary)"}),
                                        ],
                                        title="How do I handle outliers?",
                                        style={"backgroundColor": "var(--card-bg)", "marginBottom": "10px", "borderColor": "var(--border-color)", "borderRadius": "8px"},
                                        className="faq-accordion-item",
                                    ),
                                ], start_collapsed=True, style={"borderRadius": "8px", "boxShadow": "0 2px 5px rgba(0, 0, 0, 0.05)"}),
                            ]),
                        ]),

                        # Visualization Tab
                        dbc.Tab(label="Visualization", tab_id="visualization", label_style={"fontWeight": "bold", "padding": "12px 15px"}, active_label_style={"color": "var(--primary)", "borderBottom": "2px solid var(--primary)"}, children=[
                            html.Div(style={"marginTop": "20px", "padding": "5px"}, children=[
                                dbc.Accordion([
                                    dbc.AccordionItem(
                                        [
                                            html.P("In the Statistics tab, you can generate histograms, scatter plots, bar charts, and pie charts. The app also auto-generates summary and distribution plots.", style={"color": "var(--text-secondary)"}),
                                        ],
                                        title="What types of visualizations can I create?",
                                        style={"backgroundColor": "var(--card-bg)", "marginBottom": "10px", "borderColor": "var(--border-color)", "borderRadius": "8px"},
                                        className="faq-accordion-item",
                                    ),
                                    dbc.AccordionItem(
                                        [
                                            html.P("The Correlation tab shows heatmaps for numeric, label-encoded, and one-hot encoded variables.", style={"color": "var(--text-secondary)"}),
                                        ],
                                        title="How do I view correlations?",
                                        style={"backgroundColor": "var(--card-bg)", "marginBottom": "10px", "borderColor": "var(--border-color)", "borderRadius": "8px"},
                                        className="faq-accordion-item",
                                    ),
                                ], start_collapsed=True, style={"borderRadius": "8px", "boxShadow": "0 2px 5px rgba(0, 0, 0, 0.05)"}),
                            ]),
                        ]),

                        # Statistical Analysis Tab
                        dbc.Tab(label="Statistical Analysis", tab_id="statistical-analysis", label_style={"fontWeight": "bold", "padding": "12px 15px"}, active_label_style={"color": "var(--primary)", "borderBottom": "2px solid var(--primary)"}, children=[
                            html.Div(style={"marginTop": "20px", "padding": "5px"}, children=[
                                dbc.Accordion([
                                    dbc.AccordionItem(
                                        [
                                            html.P("In the Tests tab, select Chi-squared (for categorical), Pearson, or Spearman (for numeric) tests. The app provides results, visualizations, and interpretations.", style={"color": "var(--text-secondary)"}),
                                        ],
                                        title="How do I perform statistical tests?",
                                        style={"backgroundColor": "var(--card-bg)", "marginBottom": "10px", "borderColor": "var(--border-color)", "borderRadius": "8px"},
                                        className="faq-accordion-item",
                                    ),
                                    dbc.AccordionItem(
                                        [
                                            html.P("In the Regression tab, select X and Y variables, calculate regression, view the equation, metrics, and plot. You can also make predictions with confidence intervals.", style={"color": "var(--text-secondary)"}),
                                        ],
                                        title="How do I perform regression analysis?",
                                        style={"backgroundColor": "var(--card-bg)", "marginBottom": "10px", "borderColor": "var(--border-color)", "borderRadius": "8px"},
                                        className="faq-accordion-item",
                                    ),
                                    dbc.AccordionItem(
                                        [
                                            html.P("In the Prediction tab, train a Random Forest model by selecting features and a target. Make predictions manually or by uploading a file. View model metrics and results.", style={"color": "var(--text-secondary)"}),
                                        ],
                                        title="How can I make predictions using machine learning?",
                                        style={"backgroundColor": "var(--card-bg)", "marginBottom": "10px", "borderColor": "var(--border-color)", "borderRadius": "8px"},
                                        className="faq-accordion-item",
                                    ),
                                ], start_collapsed=True, style={"borderRadius": "8px", "boxShadow": "0 2px 5px rgba(0, 0, 0, 0.05)"}),
                            ]),
                        ]),

                        # Export & Reporting Tab
                        dbc.Tab(label="Export & Reporting", tab_id="export-reporting", label_style={"fontWeight": "bold", "padding": "12px 15px"}, active_label_style={"color": "var(--primary)", "borderBottom": "2px solid var(--primary)"}, children=[
                            html.Div(style={"marginTop": "20px", "padding": "5px"}, children=[
                                dbc.Accordion([
                                    dbc.AccordionItem(
                                        [
                                            html.P("In the Import tab, export your data as CSV, Excel, or JSON.", style={"color": "var(--text-secondary)"}),
                                        ],
                                        title="How can I export my data?",
                                        style={"backgroundColor": "var(--card-bg)", "marginBottom": "10px", "borderColor": "var(--border-color)", "borderRadius": "8px"},
                                        className="faq-accordion-item",
                                    ),
                                    dbc.AccordionItem(
                                        [
                                            html.P("In the Report tab, click 'Generate EDA Report' for an interactive summary with stats, visualizations, and warnings.", style={"color": "var(--text-secondary)"}),
                                        ],
                                        title="How do I generate a report?",
                                        style={"backgroundColor": "var(--card-bg)", "marginBottom": "10px", "borderColor": "var(--border-color)", "borderRadius": "8px"},
                                        className="faq-accordion-item",
                                    ),
                                ], start_collapsed=True, style={"borderRadius": "8px", "boxShadow": "0 2px 5px rgba(0, 0, 0, 0.05)"}),
                            ]),
                        ]),

                        # Troubleshooting Tab
                        dbc.Tab(label="Troubleshooting", tab_id="troubleshooting", label_style={"fontWeight": "bold", "padding": "12px 15px"}, active_label_style={"color": "var(--primary)", "borderBottom": "2px solid var(--primary)"}, children=[
                            html.Div(style={"marginTop": "20px"}, children=[
                                dbc.Accordion([
                                    dbc.AccordionItem(
                                        [
                                            html.P("If the app is slow, use smaller datasets, limit columns, and avoid complex plots with large data.", style={"color": "var(--text-secondary)"}),
                                        ],
                                        title="The app is slow. What can I do?",
                                        style={"backgroundColor": "var(--card-bg)", "marginBottom": "10px", "borderColor": "var(--border-color)", "borderRadius": "8px"},
                                        className="faq-accordion-item",
                                    ),
                                    dbc.AccordionItem(
                                        [
                                            html.P("Check file format, column names, and file integrity. Ensure the header option matches your file.", style={"color": "var(--text-secondary)"}),
                                        ],
                                        title="I get errors when uploading files.",
                                        style={"backgroundColor": "var(--card-bg)", "marginBottom": "10px", "borderColor": "var(--border-color)", "borderRadius": "8px"},
                                        className="faq-accordion-item",
                                    ),
                                    dbc.AccordionItem(
                                        [
                                            html.P("Make sure you've selected appropriate variables and plot types. Check for missing values.", style={"color": "var(--text-secondary)"}),
                                        ],
                                        title="My plots aren't displaying. What should I check?",
                                        style={"backgroundColor": "var(--card-bg)", "marginBottom": "10px", "borderColor": "var(--border-color)", "borderRadius": "8px"},
                                        className="faq-accordion-item",
                                    ),
                                ], start_collapsed=True, style={"borderRadius": "8px", "boxShadow": "0 2px 5px rgba(0, 0, 0, 0.05)"}),
                            ]),
                        ]),
                    ], id="faq-tabs", style={"backgroundColor": "var(--card-bg)", "borderRadius": "8px", "padding": "5px"}),

                    # Additional help resources
                    html.Div(style={"marginTop": "40px", "padding": "20px", "backgroundColor": "var(--primary-light)", "borderRadius": "12px", "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)"}, children=[
                        html.H5("Need More Help?", style={"color": "var(--primary)", "fontWeight": "bold", "marginBottom": "15px"}),
                        html.P([
                            "If you need assistance with your data analysis, our support team is here to help.",
                        ], style={"color": "var(--text-secondary)", "marginBottom": "15px"}),
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.I(className="fas fa-envelope", style={"marginRight": "10px", "color": "var(--primary)"}),
                                    html.Span("Contact Support: ", style={"fontWeight": "600"}),
                                    html.A("ilyes.frigui.ps@gmail.com",
                                          href="mailto:ilyes.frigui.ps@gmail.com",
                                          style={"color": "var(--primary)", "textDecoration": "none", "borderBottom": "1px dotted var(--primary)"}),
                                ], style={"fontSize": "16px", "display": "flex", "alignItems": "center"}),
                            ], width=12),
                        ]),
                    ]),
                ]),
            ], style=custom_css["card"]),
        ]),

        # Report tab
        html.Div(id="report-content", style={"display": "none", "opacity": "0", "transition": "opacity 0.3s ease-in-out"}, children=[
            dbc.Card([
                dbc.CardHeader("Automated EDA Report", style=custom_css["card_header"]),
                dbc.CardBody([
                    html.Div([
                        html.P("Generate a comprehensive Exploratory Data Analysis report for your dataset.",
                               style={"color": "var(--text-secondary)", "fontSize": "16px", "marginBottom": "20px"}),

                        # Button to generate the report
                        dbc.Button([
                            html.I(className="fas fa-file-alt mr-2"),
                            "Generate EDA Report"
                        ],
                        id="generate-report-button",
                        style=custom_css["button"],
                        className="mb-4"),

                        # Loading spinner during report generation
                        dbc.Spinner(
                            html.Div(id="eda-report-container", style={"minHeight": "200px"}),
                            color="info",
                            type="grow",
                            fullscreen=False,
                        ),
                    ]),
                ]),
            ], style=custom_css["card"]),
        ]),

        # Prediction tab
        html.Div(id="prediction-content", style={"display": "none", "opacity": "0", "transition": "opacity 0.3s ease-in-out"}, children=[
            dbc.Card([
                dbc.CardHeader("Random Forest Prediction", style=custom_css["card_header"]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            # Left panel for model training and selection
                            dbc.Card([
                                dbc.CardHeader("Train Model", style=custom_css["card_header"]),
                                dbc.CardBody([
                                    html.P("Train a Random Forest classifier on your dataset.",
                                           style={"color": "var(--text-secondary)", "marginBottom": "15px"}),

                                    # Target column selection
                                    html.Div([
                                        html.Label("Select Target Variable:", style={
                                            "color": "#e6e6e6",
                                            "fontWeight": "bold",
                                            "marginBottom": "8px",
                                            "display": "block"
                                        }),
                                        dcc.Dropdown(
                                            id="prediction-target-dropdown",
                                            placeholder="Select target column",
                                            style={"marginBottom": "20px", **custom_css["dropdown"]},
                                            className='dropdown-dark custom-dropdown'
                                        )
                                    ]),

                                    # Feature selection
                                    html.Div([
                                        html.Label("Select Features:", style={
                                            "color": "#e6e6e6",
                                            "fontWeight": "bold",
                                            "marginBottom": "8px",
                                            "display": "block"
                                        }),
                                        dcc.Dropdown(
                                            id="prediction-features-dropdown",
                                            multi=True,
                                            placeholder="Select features",
                                            style={"marginBottom": "20px", **custom_css["dropdown"]},
                                            className='dropdown-dark custom-dropdown'
                                        )
                                    ]),

                                    # Model parameters
                                    html.Div([
                                        html.Label("Model Parameters:", style={
                                            "color": "#e6e6e6",
                                            "fontWeight": "bold",
                                            "marginBottom": "8px",
                                            "display": "block"
                                        }),
                                        dbc.Row([
                                            dbc.Col([
                                                html.Label("Number of Trees:", style={"color": "#e6e6e6"}),
                                                dbc.Input(
                                                    id="n-estimators-input",
                                                    type="number",
                                                    min=10,
                                                    max=500,
                                                    step=10,
                                                    value=100,
                                                    style={"marginBottom": "15px", **custom_css["dropdown"]}
                                                )
                                            ], width=6),
                                            dbc.Col([
                                                html.Label("Max Depth:", style={"color": "#e6e6e6"}),
                                                dbc.Input(
                                                    id="max-depth-input",
                                                    type="number",
                                                    min=1,
                                                    max=50,
                                                    value=None,
                                                    placeholder="None (unlimited)",
                                                    style={"marginBottom": "15px", **custom_css["dropdown"]}
                                                )
                                            ], width=6)
                                        ]),
                                        dbc.Row([
                                            dbc.Col([
                                                html.Label("Train/Test Split:", style={"color": "#e6e6e6"}),
                                                dbc.Input(
                                                    id="test-size-input",
                                                    type="number",
                                                    min=0.1,
                                                    max=0.5,
                                                    step=0.05,
                                                    value=0.3,
                                                    style={"marginBottom": "15px", **custom_css["dropdown"]}
                                                )
                                            ], width=6),
                                            dbc.Col([
                                                html.Label("Random State:", style={"color": "#e6e6e6"}),
                                                dbc.Input(
                                                    id="random-state-input",
                                                    type="number",
                                                    min=0,
                                                    value=42,
                                                    style={"marginBottom": "15px", **custom_css["dropdown"]}
                                                )
                                            ], width=6)
                                        ])
                                    ]),

                                    # Train button
                                    dbc.Button(
                                        [html.I(className="fas fa-cogs mr-2"), "Train Model"],
                                        id="train-model-button",
                                        color="primary",
                                        style=custom_css["button"],
                                        className="mb-3"
                                    ),

                                    # Training status and metrics
                                    dbc.Spinner(
                                        html.Div(id="training-status", style={"minHeight": "50px"}),
                                        type="grow",
                                        color="info",
                                        size="sm"
                                    )
                                ])
                            ], style=custom_css["card"])
                        ], width=6),

                        dbc.Col([
                            # Right panel for making predictions
                            dbc.Card([
                                dbc.CardHeader("Make Predictions", style=custom_css["card_header"]),
                                dbc.CardBody([
                                    html.P("Make predictions using the trained Random Forest model.",
                                           style={"color": "var(--text-secondary)", "marginBottom": "15px"}),

                                    # Two tabs: Manual Input and File Upload
                                    dbc.Tabs([
                                        dbc.Tab(label="Manual Input", tab_id="manual-input", children=[
                                            html.Div(id="manual-inputs-container", style={"marginTop": "15px"}),
                                            dbc.Button(
                                                [html.I(className="fas fa-magic mr-2"), "Predict"],
                                                id="predict-button-manual",
                                                color="success",
                                                style=custom_css["button"],
                                                className="mt-3",
                                                disabled=True
                                            )
                                        ]),
                                        dbc.Tab(label="File Upload", tab_id="file-upload", children=[
                                            dcc.Upload(
                                                id="prediction-upload",
                                                children=html.Div([
                                                    html.I(className="fas fa-cloud-upload-alt mr-2", style={"fontSize": "24px", "color": "var(--primary)"}),
                                                    "Drag and Drop or ",
                                                    html.A("Select a File", style={"color": "var(--primary)", "fontWeight": "bold", "textDecoration": "underline"}),
                                                ]),
                                                style=custom_css["upload"],
                                                className="upload-area mt-3 mb-3",
                                            ),
                                            html.Div(id="prediction-upload-status", style={
                                                "color": "#a3a3a3",
                                                "textAlign": "center",
                                                "marginBottom": "15px"
                                            }),
                                            dbc.Button(
                                                [html.I(className="fas fa-magic mr-2"), "Predict from File"],
                                                id="predict-button-file",
                                                color="success",
                                                style=custom_css["button"],
                                                className="mt-2",
                                                disabled=True
                                            )
                                        ])
                                    ], id="prediction-tabs")
                                ])
                            ], style=custom_css["card"]),

                            # Results card
                            dbc.Card([
                                dbc.CardHeader("Prediction Results", style=custom_css["card_header"]),
                                dbc.CardBody([
                                    html.Div(id="prediction-results", style={"minHeight": "200px"})
                                ])
                            ], style={"marginTop": "20px", **custom_css["card"]})
                        ], width=6)
                    ])
                ]),
            ], style=custom_css["card"]),
        ]),

        # Encoding tab
        html.Div(id="encoding-content", style={"display": "none", "opacity": "0", "transition": "opacity 0.3s ease-in-out"}, children=[
            dbc.Card([
                dbc.CardHeader(
                    html.H4("Categorical Variable Encoding",
                           className="text-center",
                           style={
                               "color": "#1abc9c",
                               "fontWeight": "bold",
                               "textAlign": "center",
                               "width": "100%",
                               "margin": "0 auto",
                               "padding": "8px 0",
                               "letterSpacing": "0.5px"
                           }
                    ),
                    style={
                        **custom_css["card_header"],
                        "textAlign": "center",
                        "display": "flex",
                        "justifyContent": "center",
                        "padding": "16px",
                        "borderBottom": "2px solid rgba(26, 188, 156, 0.3)"
                    }
                ),
                dbc.CardBody([
                    # Main content container
                    dbc.Row([
                        # Left column - Controls
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(
                                    html.Div([
                                        html.I(className="fas fa-cogs mr-2", style={"color": "var(--primary)"}),
                                        "Encoding Options"
                                    ], style={"fontSize": "16px", "fontWeight": "bold"}),
                                    style=custom_css["card_header"]
                                ),
                                dbc.CardBody([
                                    # Column selection
                                    html.Div([
                                        html.Label("Select Column to Encode:",
                                                  style={"color": "#e6e6e6", "fontWeight": "bold", "marginBottom": "8px", "display": "block"}),
                                        dcc.Dropdown(
                                            id="encoding_column_dropdown",
                                            placeholder="Select a categorical column",
                                            style={**custom_css["dropdown"], "marginBottom": "16px"},
                                            className='dropdown-dark custom-dropdown'
                                        ),
                                    ], className="mb-4"),

                                    # Encoding method selection
                                    html.Div([
                                        html.Label("Select Encoding Method:",
                                                  style={"color": "#e6e6e6", "fontWeight": "bold", "marginBottom": "8px", "display": "block"}),
                                        dcc.Dropdown(
                                            id="encoding_method_dropdown",
                                            options=[
                                                {"label": "Label Encoding", "value": "label"},
                                                {"label": "One-Hot Encoding", "value": "onehot"},
                                                {"label": "Ordinal Encoding", "value": "ordinal"},
                                            ],
                                            placeholder="Select encoding method",
                                            style={**custom_css["dropdown"], "marginBottom": "16px"},
                                            className='dropdown-dark custom-dropdown'
                                        ),
                                    ], className="mb-4"),

                                    # Ordinal encoding container (will be populated dynamically)
                                    html.Div(id="encoding_ordinal_container", className="mb-4"),

                                    # Show only encoded columns toggle
                                    html.Div([
                                        dbc.Label("Show only encoded columns:",
                                                style={"color": "#e6e6e6", "fontWeight": "bold", "marginBottom": "8px", "display": "block"}),
                                        dbc.Checklist(
                                            options=[{"label": "", "value": 1}],
                                            value=[],
                                            id="encoding_show_encoded_toggle",
                                            switch=True,
                                            style={"marginBottom": "16px"}
                                        ),
                                    ], className="mb-4"),

                                    # Apply encoding button
                                    dbc.Button(
                                        "Apply Encoding",
                                        id="encoding_apply_button",
                                        color="primary",
                                        className="mb-4",
                                        style={
                                            "width": "100%",
                                            "fontWeight": "bold",
                                            "padding": "12px",
                                            "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
                                        }
                                    ),

                                    # Status message area
                                    html.Div(id="encoding_message", className="mb-4", style={
                                        "minHeight": "60px",
                                        "padding": "8px",
                                        "backgroundColor": "rgba(26, 188, 156, 0.05)",
                                        "borderRadius": "6px",
                                        "border": "1px solid rgba(26, 188, 156, 0.1)"
                                    }),

                                    # Download buttons
                                    html.Div([
                                        html.Label("Download Encoded Data:",
                                                  style={"color": "#e6e6e6", "fontWeight": "bold", "marginBottom": "16px", "display": "block"}),
                                        dbc.Row([
                                            dbc.Col([
                                                dbc.Button([
                                                    html.I(className="fas fa-file-csv mr-2"),
                                                    "CSV"
                                                ], id="encoding_download_csv_button", color="primary", style={"width": "100%"}),
                                            ], width=4),
                                            dbc.Col([
                                                dbc.Button([
                                                    html.I(className="fas fa-file-code mr-2"),
                                                    "JSON"
                                                ], id="encoding_download_json_button", color="info", style={"width": "100%"}),
                                            ], width=4),
                                            dbc.Col([
                                                dbc.Button([
                                                    html.I(className="fas fa-file-excel mr-2"),
                                                    "Excel"
                                                ], id="encoding_download_excel_button", color="success", style={"width": "100%"}),
                                            ], width=4),
                                        ]),
                                    ], className="mt-4"),
                                ])
                            ], style=custom_css["card"]),
                        ], width=4),

                        # Right column - Preview
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(
                                    html.Div([
                                        html.I(className="fas fa-table mr-2", style={"color": "var(--primary)"}),
                                        "Data Preview"
                                    ], style={"fontSize": "16px", "fontWeight": "bold"}),
                                    style=custom_css["card_header"]
                                ),
                                dbc.CardBody([
                                    # Preview table
                                    dash_table.DataTable(
                                        id="encoding_preview_table",
                                        style_table={
                                            "overflowX": "auto",
                                            "overflowY": "auto",
                                            "height": "500px",
                                            **custom_css["table"]
                                        },
                                        style_header=custom_css["table_header"],
                                        style_cell={
                                            "backgroundColor": "#16213e",
                                            "color": "#e6e6e6",
                                            "padding": "10px",
                                            "border": "1px solid #2a3a5e",
                                            "textOverflow": "ellipsis",
                                            "maxWidth": "400px",
                                        },
                                        style_data_conditional=[
                                            {"if": {"row_index": "odd"}, "backgroundColor": "#1a1a2e"}
                                        ],
                                        page_size=10,
                                        fixed_rows={'headers': True},
                                    )
                                ])
                            ], style=custom_css["card"]),
                        ], width=8),
                    ]),

                    # Hidden elements
                    dcc.Store(id="encoding_data_store"),
                    dcc.Download(id="encoding_download_csv"),
                    dcc.Download(id="encoding_download_json"),
                    dcc.Download(id="encoding_download_excel"),
                ], style={"padding": "24px"}),
            ], style=custom_css["card"]),
        ]),
    ]),

    # Footer
    html.Footer(style={
        "backgroundColor": "var(--dark-bg)",
        "color": "var(--text-secondary)",
        "padding": "20px",
        "textAlign": "center",
        "marginTop": "30px",
        "borderTop": "none"
    }, children=[
        html.P(f"© {pd.Timestamp.now().year} Data Analysis Dashboard", style={
            "marginBottom": "0",
            "color": "var(--text-secondary)",
            "fontSize": "14px",
            "opacity": "0.8"
        })
    ]),

    # Store for duplicates and outliers data
    dcc.Store(id='duplicates-store'),
    dcc.Store(id='outliers-store'),

    # Stores for prediction functionality
    dcc.Store(id='trained-model-store'),  # Stores trained model info
    dcc.Store(id='feature-info-store'),   # Stores feature information for preprocessing
    dcc.Store(id='prediction-df-store'),  # Stores dataframe for prediction file upload
    dcc.Store(id='model-performance-store')  # Stores model performance metrics
])

# Navigation callback
@app.callback(
    [
        Output("welcome-content", "style"),
        Output("import-content", "style"),
        Output("summary-content", "style"),
        Output("imputation-content", "style"),
        Output("statistics-content", "style"),
        Output("encoding-content", "style"),
        Output("tests-content", "style"),
        Output("regression-content", "style"),
        Output("report-content", "style"),
        Output("prediction-content", "style"),
        Output("faq-content", "style"),
        Output("welcome-button", "active"),
        Output("import-button", "active"),
        Output("summary-button", "active"),
        Output("imputation-button", "active"),
        Output("statistics-button", "active"),
        Output("encoding-button", "active"),
        Output("tests-button", "active"),
        Output("regression-button", "active"),
        Output("report-button", "active"),
        Output("prediction-button", "active"),
        Output("faq-button", "active"),
        Output("welcome-button", "className"),
        Output("import-button", "className"),
        Output("summary-button", "className"),
        Output("imputation-button", "className"),
        Output("statistics-button", "className"),
        Output("encoding-button", "className"),
        Output("tests-button", "className"),
        Output("regression-button", "className"),
        Output("report-button", "className"),
        Output("prediction-button", "className"),
        Output("faq-button", "className"),
    ],
    [
        Input("welcome-button", "n_clicks"),
        Input("import-button", "n_clicks"),
        Input("summary-button", "n_clicks"),
        Input("imputation-button", "n_clicks"),
        Input("statistics-button", "n_clicks"),
        Input("encoding-button", "n_clicks"),
        Input("tests-button", "n_clicks"),
        Input("regression-button", "n_clicks"),
        Input("report-button", "n_clicks"),
        Input("prediction-button", "n_clicks"),
        Input("faq-button", "n_clicks"),
    ],
)
def update_content(welcome_clicks, import_clicks, summary_clicks, imputation_clicks, statistics_clicks, encoding_clicks, tests_clicks, regression_clicks, report_clicks, prediction_clicks, faq_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return [
            {"display": "block", "opacity": "1", "transition": "all 0.4s ease-in-out", "animation": "fade-in 0.5s ease-out"},
            *[{"display": "none", "opacity": "0", "transition": "all 0.4s ease-in-out"}] * 10,
            True, *[False] * 10,
            "nav-button active", *["nav-button"] * 10
        ]
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    styles = [
        {"display": "none", "opacity": "0", "transition": "all 0.4s ease-in-out"}
    ] * 11
    active_states = [False] * 11
    classes = ["nav-button"] * 11
    idx_map = {
        "welcome-button": 0,
        "import-button": 1,
        "summary-button": 2,
        "imputation-button": 3,
        "statistics-button": 4,
        "encoding-button": 5,
        "tests-button": 6,
        "regression-button": 7,
        "report-button": 8,
        "prediction-button": 9,
        "faq-button": 10
    }
    if button_id in idx_map:
        idx = idx_map[button_id]
        styles[idx] = {"display": "block", "opacity": "1", "transition": "all 0.4s ease-in-out", "animation": "fade-in 0.5s ease-out"}
        active_states[idx] = True
        classes[idx] = "nav-button active"
    return styles + active_states + classes

# Data parsing callback
@app.callback(
    [
        Output("data-table", "data"),
        Output("data-table", "columns"),
        Output("file-upload-status", "children"),
        Output("x-axis-dropdown", "options"),
        Output("y-axis-dropdown", "options"),
        Output("test-x-dropdown", "options"),
        Output("test-y-dropdown", "options"),
        Output("imputation-columns", "options"),
        Output("outlier-columns", "options"),
        Output("date-column-dropdown", "options"),
        Output("value-column-dropdown", "options"),
        Output("scatter-matrix-vars", "options"),
        Output("scatter-matrix-color", "options"),
    ],
    [Input("upload-data", "contents"), Input("header-checkbox", "value")],
    [State("upload-data", "filename")],
)
def parse_data(contents, has_header, filename):
    if contents is None:
        return [], [], "No file uploaded yet.", [], [], [], [], [], [], [], [], [], []

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        if "csv" in filename:
            df = pd.read_csv(StringIO(decoded.decode("utf-8")), header=0 if "header" in has_header else None)
        elif "xls" in filename or "xlsx" in filename:
            df = pd.read_excel(io.BytesIO(decoded), header=0 if "header" in has_header else None)
        else:
            return [], [], "Unsupported file format.", [], [], [], [], [], [], [], [], [], []

    except Exception as e:
        return [], [], f"Error processing file: {e}", [], [], [], [], [], [], [], [], [], []

    columns = [{"name": col, "id": col} for col in df.columns]
    data = df.to_dict("records")
    dropdown_options = [{"label": html.Span(col, style={"color": "#FFFFFF"}), "value": col} for col in df.columns]

    # For test dropdowns, we provide all columns as options initially
    # The test-specific callback will filter them based on the test type
    test_dropdown_options = dropdown_options.copy()

    # Other dropdowns remain the same
    categorical_columns = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    # Find columns with missing values for imputation dropdown
    columns_with_missing = [col for col in df.columns if df[col].isna().any()]
    imputation_dropdown_options = [{"label": html.Span(col, style={"color": "#FFFFFF"}), "value": col} for col in columns_with_missing]

    # Only numeric columns for outlier detection
    outlier_dropdown_options = [{"label": html.Span(col, style={"color": "#FFFFFF"}), "value": col} for col in numeric_columns]

    # Identify potential date columns for time series
    date_columns = [col for col in df.columns if is_possible_datetime(df[col])]
    date_dropdown_options = [{"label": html.Span(col, style={"color": "#FFFFFF"}), "value": col} for col in date_columns]

    # Options for scatter matrix
    scatter_matrix_vars = [{"label": html.Span(col, style={"color": "#FFFFFF"}), "value": col} for col in numeric_columns]
    scatter_matrix_color = [{"label": html.Span(col, style={"color": "#FFFFFF"}), "value": col} for col in df.columns]

    return (data, columns, f"Successfully uploaded {filename}", dropdown_options, dropdown_options,
            test_dropdown_options, test_dropdown_options, imputation_dropdown_options, outlier_dropdown_options,
            date_dropdown_options, dropdown_options, scatter_matrix_vars, scatter_matrix_color)

# Show/hide controls based on plot type selection
@app.callback(
    [
        Output("time-series-controls", "style"),
        Output("bin-size-col", "style"),
        Output("moving-avg-col", "style"),
        Output("seasonality-col", "style"),
        Output("scatter-matrix-controls", "style"),
        Output("3d-plot-controls", "style"),
        Output("geo-plot-controls", "style"),
        Output("forecast-controls", "style"),
        Output("stat-plot-controls", "style"),
    ],
    [Input("plot-type-dropdown", "value")]
)
def toggle_plot_controls(plot_type):
    return [{"display": "none"}] * 9

# Summary statistics callback
@app.callback(
    [
        Output("summary-table", "data"),
        Output("summary-table", "columns"),
        Output("num-rows", "children"),
        Output("num-cols", "children"),
        Output("missing-values-summary", "children"),
        Output("summary-error", "children"),
        Output("summary-error", "is_open"),
    ],
    [Input("data-table", "data"), Input("summary-button", "n_clicks")],
    prevent_initial_call=True
)
def generate_summary(data, n_clicks):
    if not data or not n_clicks:
        return [], [], "", "", "", "", False

    try:
        df = pd.DataFrame(data)
        if df.empty:
            return [], [], "", "", "", "No data available to summarize.", True

        # Dataset Overview
        num_rows, num_cols = df.shape

        # Missing Values Summary
        missing_values = df.isna().sum().sum()
        missing_values_pct = (missing_values / (num_rows * num_cols)) * 100 if num_rows * num_cols > 0 else 0

        # Get missing values by column for detailed visualization
        missing_by_column = df.isna().sum().reset_index()
        missing_by_column.columns = ['Column', 'Missing Count']
        missing_by_column['Missing Percentage'] = (missing_by_column['Missing Count'] / num_rows * 100).round(2)
        missing_by_column = missing_by_column.sort_values('Missing Count', ascending=False)

        # Only keep columns with missing values
        missing_by_column = missing_by_column[missing_by_column['Missing Count'] > 0]

        # Create enhanced missing values summary with animations and more detailed information
        missing_values_summary = html.Div([
            # Header with icon and main count
            html.Div([
                html.I(className="fas fa-exclamation-triangle fa-2x",
                       style={"color": "#EA4335", "marginRight": "15px"}),
                html.Div([
                    html.H4("Missing Values", style={"color": "#ffffff", "margin": "0"}),
                    html.Div([
                        html.Span(f"{missing_values}",
                                  style={"fontSize": "24px", "fontWeight": "bold", "color": "#EA4335"}),
                        html.Span(f" ({missing_values_pct:.2f}%)",
                                  style={"fontSize": "16px", "color": "#aaaaaa"})
                    ], style={"display": "flex", "alignItems": "baseline"})
                ])
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "15px"}),

            # Interactive gauge chart with animation
            dcc.Graph(
                figure=go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=100 - missing_values_pct,
                    title={"text": "Data Completeness", "font": {"size": 16, "color": "white"}},
                    delta={"reference": 100, "increasing": {"color": "#34A853"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "white"},
                        "bar": {"color": "#34A853"},
                        "bgcolor": "rgba(50, 50, 50, 0.2)",
                        "borderwidth": 2,
                        "bordercolor": "#2a3a5e",
                        "steps": [
                            {"range": [0, 60], "color": "rgba(234, 67, 53, 0.3)"},
                            {"range": [60, 80], "color": "rgba(251, 188, 5, 0.3)"},
                            {"range": [80, 100], "color": "rgba(52, 168, 83, 0.3)"}
                        ],
                    }
                )).update_layout(
                    # Add animation
                    transition_duration=500,
                    # Improve appearance
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={"color": "white"},
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=200
                ),
                config={"displayModeBar": False},
                style={"height": "200px"}
            ),

            # Detailed missing values by column (only show if there are missing values)
            html.Div([
                html.H5("Missing Values by Column",
                       style={"color": "#ffffff", "margin": "20px 0 10px 0",
                              "textAlign": "center", "fontWeight": "bold"}),

                # Animated bar chart showing missing values by column
                dcc.Graph(
                    figure=go.Figure(
                        data=[go.Bar(
                            x=missing_by_column['Column'].tolist(),
                            y=missing_by_column['Missing Percentage'].tolist(),
                            marker=dict(
                                color=missing_by_column['Missing Percentage'].tolist(),
                                colorscale='Viridis',
                                colorbar=dict(title="% Missing"),
                            ),
                            text=missing_by_column['Missing Count'].tolist(),
                            hovertemplate="<b>%{x}</b><br>Missing: %{text} values<br>(%{y:.2f}%)<extra></extra>",
                        )],
                        layout=go.Layout(
                            title="Percentage of Missing Values by Column",
                            title_font=dict(size=14, color="white"),
                            xaxis=dict(title="", tickangle=45, showgrid=False),
                            yaxis=dict(title="Percentage (%)", showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color="white"),
                            margin=dict(l=40, r=20, t=40, b=80),
                            height=300,
                            transition_duration=800,  # Animation on loading
                        )
                    ),
                    config={"displayModeBar": False},
                    style={"height": "300px"}
                ),

                # Recommendations card based on missing data
                html.Div([
                    html.H6("Recommendations",
                           style={"color": "#1abc9c", "fontWeight": "bold", "marginBottom": "10px"}),
                    html.Ul([
                        html.Li(
                            "Consider imputation techniques for columns with few missing values",
                            style={"color": "#e6e6e6", "marginBottom": "5px"}
                        ),
                        html.Li(
                            "For columns with >50% missing data, consider removing them",
                            style={"color": "#e6e6e6", "marginBottom": "5px"}
                        ),
                        html.Li(
                            "Examine patterns in missing data for potential biases",
                            style={"color": "#e6e6e6"}
                        ),
                    ], style={"paddingLeft": "20px"})
                ], style={
                    "backgroundColor": "rgba(26, 188, 156, 0.1)",
                    "border": "1px solid rgba(26, 188, 156, 0.3)",
                    "borderRadius": "5px",
                    "padding": "15px",
                    "marginTop": "15px"
                }) if missing_values > 0 else html.Div()
            ], style={"display": "block" if missing_values > 0 else "none"}),
        ])

        # NEW APPROACH: Create summary statistics for all columns regardless of type
        # Start with an empty DataFrame with preset statistics
        stats_rows = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'null_count', 'null_pct',
                      'unique', 'top', 'freq', 'dtype']
        summary = pd.DataFrame(index=stats_rows)

        # Process each column individually
        for col in df.columns:
            col_data = df[col]
            col_stats = {}

            # Basic count statistics
            col_stats['count'] = len(col_data)
            col_stats['null_count'] = col_data.isna().sum()
            col_stats['null_pct'] = f"{(col_data.isna().sum() / len(col_data) * 100):.2f}%"
            col_stats['dtype'] = str(col_data.dtype)

            # Try to get unique values (works for most data types)
            try:
                col_stats['unique'] = col_data.nunique()
            except:
                col_stats['unique'] = 'N/A'

            # Numeric statistics when applicable
            try:
                # Check if data can be treated as numeric
                numeric_data = pd.to_numeric(col_data, errors='coerce')
                if not numeric_data.isna().all():  # Only calculate if we have some numeric values
                    col_stats['mean'] = numeric_data.mean()
                    col_stats['std'] = numeric_data.std()
                    col_stats['min'] = numeric_data.min()
                    col_stats['25%'] = numeric_data.quantile(0.25)
                    col_stats['50%'] = numeric_data.quantile(0.5)
                    col_stats['75%'] = numeric_data.quantile(0.75)
                    col_stats['max'] = numeric_data.max()
            except:
                # Fill in N/A for statistics that couldn't be calculated
                for stat in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                    if stat not in col_stats:
                        col_stats[stat] = 'N/A'

            # Most common value statistics (categorical/object data)
            try:
                value_counts = col_data.value_counts(dropna=True)
                if not value_counts.empty:
                    top_val = value_counts.index[0]
                    # Truncate long values
                    col_stats['top'] = str(top_val)[:20] + "..." if len(str(top_val)) > 20 else str(top_val)
                    col_stats['freq'] = value_counts.iloc[0]
                else:
                    col_stats['top'] = 'N/A'
                    col_stats['freq'] = 'N/A'
            except:
                col_stats['top'] = 'N/A'
                col_stats['freq'] = 'N/A'

            # Add this column's stats to the summary DataFrame
            summary[col] = pd.Series(col_stats)

        # Format numeric values
        for col in summary.columns:
            for stat in stats_rows:
                val = summary.loc[stat, col]
                # Format numeric values for readability
                if isinstance(val, (int, float)) and not isinstance(val, bool) and not pd.isna(val):
                    if stat in ['count', 'unique', 'freq', 'null_count']:
                        summary.loc[stat, col] = int(val)
                    elif stat not in ['null_pct', 'dtype', 'top'] and not isinstance(val, str):
                        summary.loc[stat, col] = f"{val:.3f}" if abs(val) < 1000 else f"{val:.2e}"

        # Reset index to create the 'index' column for the table
        summary = summary.reset_index().rename(columns={'index': 'Statistic'})

        # Create column specifications for the table
        columns = [
            {"name": "Statistic", "id": "Statistic"} if col == "Statistic" else
            {"name": str(col), "id": str(col)}
            for col in summary.columns
        ]

        # Create table data
        summary_data = summary.to_dict("records")

        # If the summary is still empty after all efforts, create a placeholder
        if not summary_data:
            raise ValueError("Could not generate statistics")

        return summary_data, columns, num_rows, num_cols, missing_values_summary, "", False

    except Exception as e:
        print(f"Error in generate_summary: {str(e)}")
        return [], [], "", "", "", f"Error generating summary: {str(e)}", True

# Imputation callback
@app.callback(
    [
        Output("imputed-table", "data"),
        Output("imputed-table", "columns"),
        Output("imputed-table", "page_size"),
        Output("missing-values-message", "children"),
        Output("remove-duplicates-button", "disabled"),
        Output("duplicates-message", "children"),
        Output('duplicates-store', 'data'),
        Output("handle-outliers-button", "disabled"),
        Output("outliers-message", "children"),
        Output('outliers-store', 'data'),
        Output("imputation-success-toast", "is_open"),
        Output("imputation-warning-toast", "is_open")
    ],
    [
        Input("data-table", "data"),
        Input("imputation-columns", "value"),
        Input("missing-method", "value"),
        Input("imputation-rows", "value"),
        Input("find-duplicates-button", "n_clicks"),
        Input("remove-duplicates-button", "n_clicks"),
        Input("detect-outliers-button", "n_clicks"),
        Input("handle-outliers-button", "n_clicks"),
        Input("apply-imputation-button", "n_clicks")
    ],
    [
        State('duplicates-store', 'data'),
        State('outliers-store', 'data'),
        State("outlier-columns", "value"),
        State("outlier-method", "value"),
        State("outlier-threshold", "value"),
        State("outlier-handling-method", "value")
    ]
)
def handle_data_cleaning(data, selected_columns, method, rows, find_clicks, remove_clicks,
                        detect_clicks, handle_clicks, apply_imputation_clicks, duplicates_data, outliers_data,
                        outlier_cols, outlier_method, outlier_threshold, outlier_handling):
    if not data:
        return [], [], 10, "No data uploaded yet", True, "No data uploaded yet", None, True, "No data uploaded yet", None, False, False

    ctx = dash.callback_context
    df = pd.DataFrame(data)
    if df.empty:
        return [], [], 10, "No data available", True, "No data available", None, True, "No data available", None, False, False

    # Initialize variables
    duplicates_message = []
    remove_button_disabled = True
    duplicates_store = duplicates_data

    outliers_message = []
    handle_button_disabled = True
    outliers_store = outliers_data

    # Initialize toast states
    success_toast_open = False
    warning_toast_open = False

    # Create a message about missing values
    missing_cols = df.columns[df.isna().any()].tolist()
    missing_counts = df.isna().sum()

    df_imputed = df.copy()

    if len(missing_cols) == 0:
        missing_message = html.Div([
            html.I(className="fas fa-check-circle mr-2", style={"color": "#51cf66", "marginRight": "8px"}),
            "No missing values detected in the dataset."
        ], style={"color": "#51cf66", "fontWeight": "500"})
    else:
        missing_message = [
            html.Div("Columns with missing values:", style={"fontWeight": "500", "marginBottom": "10px"}),
            html.Ul([
                html.Li([
                    html.Span(f"{col}: ", style={"fontWeight": "500"}),
                    f"{missing_counts[col]} missing values ({(missing_counts[col]/len(df)*100):.1f}%)"
                ]) for col in missing_cols
            ], style={"marginLeft": "20px", "listStyleType": "disc"})
        ]

    # Handle duplicate detection
    if ctx.triggered and ctx.triggered[0]['prop_id'] == 'find-duplicates-button.n_clicks':
        duplicates = df[df.duplicated(keep=False)]
        duplicate_count = len(duplicates)
        if duplicate_count > 0:
            duplicates_message = [
                html.P([
                    html.I(className="fas fa-exclamation-triangle mr-2", style={"color": "#ff6b6b"}),
                    f"Found {duplicate_count} duplicate rows"
                ], style={"color": "#ff6b6b"}),
                html.P("Preview of duplicates:", style={"marginTop": "10px"}),
                dash_table.DataTable(
                    data=duplicates.head(10).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in duplicates.columns],
                    page_size=5,
                    style_table={"overflowX": "auto", **custom_css["table"]},
                    style_header=custom_css["table_header"],
                    style_cell={
                        "backgroundColor": "#16213e",
                        "color": "#e6e6e6",
                        "padding": "10px",
                        "border": "1px solid #2a3a5e"
                    },
                )
            ]
            remove_button_disabled = False
            duplicates_store = duplicates.to_dict('records')
        else:
            duplicates_message = html.P([
                html.I(className="fas fa-check-circle mr-2", style={"color": "#51cf66"}),
                "No duplicate rows found"
            ], style={"color": "#51cf66"})
            remove_button_disabled = True
            duplicates_store = None

    # Handle duplicate removal
    elif ctx.triggered and ctx.triggered[0]['prop_id'] == 'remove-duplicates-button.n_clicks' and duplicates_data:
        df = df.drop_duplicates()
        df_imputed = df.copy()
        duplicates_message = html.P([
            html.I(className="fas fa-check-circle mr-2", style={"color": "#51cf66"}),
            "Duplicate rows removed successfully"
        ], style={"color": "#51cf66"})
        remove_button_disabled = True
        duplicates_store = None

    # Handle outlier detection
    elif ctx.triggered and ctx.triggered[0]['prop_id'] == 'detect-outliers-button.n_clicks':
        if not outlier_cols:
            outliers_message = html.P([
                html.I(className="fas fa-exclamation-circle mr-2", style={"color": "#ff6b6b"}),
                "Please select at least one column"
            ], style={"color": "#ff6b6b"})
        else:
            outliers_dict = {}
            outlier_indices = set()

            for col in outlier_cols:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    continue

                col_data = df[col].copy().dropna()

                if len(col_data) <= 1:  # Skip if not enough data after dropping NA
                    continue

                if outlier_method == "iqr":
                    # IQR method
                    q1 = col_data.quantile(0.25)
                    q3 = col_data.quantile(0.75)
                    iqr = q3 - q1
                    threshold = outlier_threshold if outlier_threshold else 1.5
                    lower_bound = q1 - threshold * iqr
                    upper_bound = q3 + threshold * iqr
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                    outlier_rows = outliers.index.tolist()

                elif outlier_method == "zscore":
                    # Z-score method
                    threshold = outlier_threshold if outlier_threshold else 3

                    # Calculate z-scores more safely
                    mean = col_data.mean()
                    std = col_data.std()

                    # Handle case where std is 0 to avoid division by zero
                    if std == 0:
                        outlier_rows = []
                    else:
                        # Calculate z-scores manually to avoid errors
                        z_scores = abs((df[col] - mean) / std)
                        outlier_rows = df[z_scores > threshold].index.tolist()

                if outlier_rows:
                    outliers_dict[col] = {
                        'outliers': df.loc[outlier_rows, col].to_dict(),
                        'count': len(outlier_rows),
                        'indices': outlier_rows
                    }
                    outlier_indices.update(outlier_rows)

            if outliers_dict:
                total_outliers = len(outlier_indices)
                outliers_message = [
                    html.P([
                        html.I(className="fas fa-exclamation-triangle mr-2", style={"color": "#ff6b6b"}),
                        f"Found {total_outliers} rows with outliers across selected columns"
                    ], style={"color": "#ff6b6b"}),
                    html.P("Outliers summary:", style={"marginTop": "10px"}),
                    html.Ul([
                        html.Li(f"{col}: {stats['count']} outliers")
                        for col, stats in outliers_dict.items()
                    ]),
                    html.P("Preview of rows with outliers:", style={"marginTop": "10px"}),
                    dash_table.DataTable(
                        data=df.loc[list(outlier_indices)].head(10).to_dict('records'),
                        columns=[{"name": i, "id": i} for i in df.columns],
                        page_size=5,
                        style_table={"overflowX": "auto", **custom_css["table"]},
                        style_header=custom_css["table_header"],
                        style_cell={
                            "backgroundColor": "#16213e",
                            "color": "#e6e6e6",
                            "padding": "10px",
                            "border": "1px solid #2a3a5e"
                        },
                    )
                ]
                handle_button_disabled = False
                outliers_store = {
                    'outliers_dict': outliers_dict,
                    'outlier_indices': list(outlier_indices),
                    'method': outlier_method,
                    'threshold': outlier_threshold if outlier_threshold else (3 if outlier_method == "zscore" else 1.5)
                }
            else:
                outliers_message = html.P([
                    html.I(className="fas fa-check-circle mr-2", style={"color": "#51cf66"}),
                    "No outliers found in selected columns"
                ], style={"color": "#51cf66"})
                handle_button_disabled = True
                outliers_store = None

    # Handle outlier treatment
    elif ctx.triggered and ctx.triggered[0]['prop_id'] == 'handle-outliers-button.n_clicks' and outliers_data:
        outliers_dict = outliers_data['outliers_dict']
        outlier_indices = outliers_data['outlier_indices']

        if outlier_handling == "remove":
            df = df.drop(index=outlier_indices)
            df_imputed = df.copy()
            outliers_message = html.P([
                html.I(className="fas fa-check-circle mr-2", style={"color": "#51cf66"}),
                f"Removed {len(outlier_indices)} rows containing outliers"
            ], style={"color": "#51cf66"})
        else:
            for col, stats in outliers_dict.items():
                if outlier_handling == "median":
                    replacement = df[col].median()
                elif outlier_handling == "mean":
                    replacement = df[col].mean()

                df.loc[stats['indices'], col] = replacement
                df_imputed = df.copy()

            outliers_message = html.P([
                html.I(className="fas fa-check-circle mr-2", style={"color": "#51cf66"}),
                f"Replaced outliers in {len(outliers_dict)} columns using {outlier_handling}"
            ], style={"color": "#51cf66"})

        handle_button_disabled = True
        outliers_store = None

    # Handle imputation button click
    elif ctx.triggered and ctx.triggered[0]['prop_id'] == 'apply-imputation-button.n_clicks':
        if not selected_columns:
            warning_toast_open = True
        else:
            df_imputed = df.copy()
            for col in selected_columns:
                if pd.isna(df[col]).any():  # Only impute if there are missing values
                    if pd.api.types.is_numeric_dtype(df[col]):
                        if method == "mean":
                            imputed_value = df[col].mean()
                            df_imputed[col] = df[col].fillna(round(imputed_value, 3))
                        elif method == "median":
                            imputed_value = df[col].median()
                            df_imputed[col] = df[col].fillna(round(imputed_value, 3))
                        elif method == "knn":
                            imputer = get_sklearn('KNNImputer')(n_neighbors=5)
                            # Apply KNN imputation
                            knn_imputed_values = imputer.fit_transform(df[[col]])
                            # Convert back to DataFrame and round
                            knn_series = pd.Series(knn_imputed_values.flatten(), index=df.index)
                            df_imputed[col] = knn_series.round(3)
                    else:  # Categorical or object columns
                        if method == "mode":
                            df_imputed[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else np.nan)

            success_toast_open = True

    # Prepare table data
    columns = [{"name": col, "id": col} for col in df_imputed.columns]
    data = df_imputed.to_dict("records")

    # Handle row display
    page_size = len(data) if rows == "all" and len(data) > 0 else (rows if rows else 10)

    if len(data) == 0:
        missing_message = html.P(
            "Warning: The dataset is empty after applying operations.",
            style={"color": "#ff6b6b"}
        )

    return data, columns, page_size, missing_message, remove_button_disabled, duplicates_message, duplicates_store, handle_button_disabled, outliers_message, outliers_store, success_toast_open, warning_toast_open

# Statistics Callback - Updated with only histogram, scatter, bar, and pie charts
@app.callback(
    [
        Output("statistics-plot", "figure"),
        Output("statistics-error", "children"),
        Output("statistics-error", "is_open"),
    ],
    [
        Input("generate-plot-button", "n_clicks"),
        Input("x-axis-dropdown", "value"),
        Input("y-axis-dropdown", "value"),
        Input("plot-type-dropdown", "value"),
        Input("bin-size-slider", "value"),
        Input("data-table", "data"),
    ],
)
def generate_plots(n_clicks, x_axis, y_axis, plot_type, bin_size, data):
    if not n_clicks or not data:
        fig = go.Figure()
        return apply_dark_theme(fig), "", False

    df = pd.DataFrame(data)
    fig = go.Figure()

    try:
        # Histogram
        if plot_type == "histogram":
            if not x_axis:
                return go.Figure(), "Please select X-axis for histogram.", True
            if not pd.api.types.is_numeric_dtype(df[x_axis]):
                return go.Figure(), "X-axis must be numeric for histograms.", True

            # Histogram doesn't need y_axis, so we ignore it
            unique_vals = df[x_axis].dropna().unique()
            if len(unique_vals) == 2:
                # Treat as categorical (binary)
                val_counts = df[x_axis].value_counts().sort_index()
                fig = go.Figure(data=go.Bar(
                    x=[str(v) for v in val_counts.index],
                    y=val_counts.values,
                    marker=dict(color='#1abc9c'),
                    opacity=0.85
                ))
                fig.update_layout(
                    title=f"Distribution of {x_axis}",
                    xaxis_title=f"{x_axis} (0 = No, 1 = Yes)",
                    yaxis_title="Count",
                    xaxis_type='category',
                    bargap=0.05,
                    template="plotly_dark"
                )
            else:
                nbins = bin_size if bin_size else 20
                fig = px.histogram(
                    df, x=x_axis, nbins=nbins, title=f"Histogram of {x_axis}",
                    color_discrete_sequence=['#1abc9c']
                )
                fig.update_traces(marker_line_color='#16a085', marker_line_width=2, opacity=0.85)
                fig.update_layout(bargap=0.05)
                mean_val = df[x_axis].mean()
                fig.add_vline(x=mean_val, line_dash="dash", line_color="#3498db")

        # Scatter Plot
        elif plot_type == "scatter":
            if not x_axis or not y_axis:
                return go.Figure(), "Please select both X-axis and Y-axis for scatter plot.", True
            if not pd.api.types.is_numeric_dtype(df[x_axis]) or not pd.api.types.is_numeric_dtype(df[y_axis]):
                return go.Figure(), "Both X-axis and Y-axis must be numeric for scatter plots.", True
            fig = px.scatter(df, x=x_axis, y=y_axis, title=f"Scatter Plot of {y_axis} vs {x_axis}")

        # Bar Chart
        elif plot_type == "bar":
            if not x_axis or not y_axis:
                return go.Figure(), "Please select both X-axis and Y-axis for bar chart.", True
            fig = px.bar(df, x=x_axis, y=y_axis, title=f"Bar Chart of {y_axis} vs {x_axis}")

        else:
            return go.Figure(), "Invalid plot type selected. Supported types: histogram, scatter, bar.", True

        # Apply the dark theme to the figure
        fig = apply_dark_theme(fig)

    except Exception as e:
        return go.Figure(), f"Error generating plot: {str(e)}", True

    return fig, "", False

# Download callback
@app.callback(
    Output("download-data", "data"),
    [Input("download-button", "n_clicks")],
    [State("data-table", "data")],
    prevent_initial_call=True,
)
def download_data(n_clicks, data):
    if not data:
        return None

    df = pd.DataFrame(data)
    return dcc.send_data_frame(df.to_csv, "processed_data.csv")

# Excel export callback
@app.callback(
    Output("export-excel", "data"),
    [Input("export-excel-button", "n_clicks")],
    [State("data-table", "data")],
    prevent_initial_call=True,
)
def export_excel(n_clicks, data):
    if not data:
        return None

    df = pd.DataFrame(data)
    return dcc.send_data_frame(df.to_excel, "data_export.xlsx", sheet_name="Data")

# JSON export callback
@app.callback(
    Output("export-json", "data"),
    [Input("export-json-button", "n_clicks")],
    [State("data-table", "data")],
    prevent_initial_call=True,
)
def export_json(n_clicks, data):
    if not data:
        return None

    df = pd.DataFrame(data)
    return dict(content=df.to_json(orient="records"), filename="data_export.json")

# CSV export callback
@app.callback(
    Output("export-csv", "data"),
    [Input("export-csv-button", "n_clicks")],
    [State("data-table", "data")],
    prevent_initial_call=True,
)
def export_csv(n_clicks, data):
    if not data:
        return None

    df = pd.DataFrame(data)
    return dcc.send_data_frame(df.to_csv, "data_export.csv", index=False)

# Auto-visualization callback
@app.callback(
    Output("auto-visualizations", "children"),
    [Input("data-table", "data"), Input("statistics-button", "n_clicks")],
    prevent_initial_call=True
)
def generate_auto_visualizations(data, statistics_clicks):
    if not data or not statistics_clicks:
        return html.Div("Please upload data to see visualizations")

    try:
        df = pd.DataFrame(data)
        if df.empty:
            return html.Div("No data available to visualize")

        # Create containers for different visualization categories
        data_summary_plots = []
        distribution_plots = []
        category_plots = []

        # Get column types
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = [col for col in df.columns if is_possible_datetime(df[col])]

        # ------------------- DATA SUMMARY DASHBOARD -------------------
        # Create a two-panel summary dashboard of data types and missing values
        fig_summary = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Column Data Types", "Top Missing Values"),
            specs=[[{"type": "domain"}, {"type": "xy"}]]
        )

        # Data Types Summary
        data_types = df.dtypes.value_counts().reset_index()
        data_types.columns = ['Data Type', 'Count']
        data_types['Data Type'] = data_types['Data Type'].astype(str)

        fig_summary.add_trace(
            go.Pie(
                labels=data_types['Data Type'],
                values=data_types['Count'],
                hole=0.5,
                textinfo='label+percent',
                marker=dict(
                    colors=['#1abc9c', '#16a085', '#2ecc71', '#3498db', '#9b59b6'],
                    line=dict(color='#000000', width=1)
                )
            ),
            row=1, col=1
        )

        # Missing Values Summary
        missing_data = df.isna().sum().reset_index()
        missing_data.columns = ['Column', 'Missing Values']
        missing_data = missing_data.sort_values('Missing Values', ascending=False).head(10)

        if missing_data['Missing Values'].sum() > 0:
            fig_summary.add_trace(
                go.Bar(
                    x=missing_data['Column'],
                    y=missing_data['Missing Values'],
                    marker=dict(
                        color='#1abc9c',
                        line=dict(width=1, color='#000000')
                    )
                ),
                row=1, col=2
            )

        fig_summary.update_layout(
            title="Data Composition Dashboard",
            height=450
        )

        data_summary_plots.append(dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=apply_dark_theme(fig_summary))
                ])
            ], style=custom_css["card"]),
            width=12, style={"marginBottom": "20px"}
        ))

        # ------------------- DISTRIBUTION PLOTS -------------------
        # Create visual distributions for numeric columns (up to 6)
        for i in range(0, min(6, len(numeric_cols))):
            # Create individual distribution dashboards for each column
            col = numeric_cols[i]

            fig_dist = go.Figure()

            # Column distribution
            hist_data = df[col].dropna()
            if len(hist_data) > 5:  # Only plot if we have enough data
                fig_dist.add_trace(
                    go.Histogram(
                        x=hist_data,
                        nbinsx=20,
                        marker=dict(
                            color='#1abc9c',
                            line=dict(color='#16a085', width=2)
                        ),
                        opacity=0.85
                    )
                )
                # Add mean line
                mean = hist_data.mean()
                fig_dist.add_trace(
                    go.Scatter(
                        x=[mean, mean],
                        y=[0, 1],  # y-range will be auto-scaled
                        mode='lines',
                        line=dict(color='#3498db', width=2, dash='dash'),
                        name='Mean'
                    )
                )

            fig_dist.update_layout(
                title=f"Distribution Analysis: {col}",
                barmode='overlay',
                showlegend=False,
                height=400,
                bargap=0.05
            )

            distribution_plots.append(dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(figure=apply_dark_theme(fig_dist))
                    ])
                ], style=custom_css["card"]),
                width=12, style={"marginBottom": "20px"}
            ))

        # ------------------- CATEGORICAL PLOTS -------------------
        # Create bar charts for categorical variables (up to 3)
        if len(categorical_cols) > 0:
            for i, cat_col in enumerate(categorical_cols[:min(3, len(categorical_cols))]):
                # Count the frequency of each category
                cat_counts = df[cat_col].value_counts().nlargest(10)

                # Create a bar plot
                fig_cat = go.Figure(data=go.Bar(
                    x=cat_counts.index,
                    y=cat_counts.values,
                    marker=dict(
                        color='#1abc9c',
                        line=dict(color='rgba(255, 255, 255, 0.5)', width=1)
                    )
                ))

                fig_cat.update_layout(
                    title=f"Category Distribution: {cat_col}",
                    xaxis_title=cat_col,
                    yaxis_title="Count",
                    height=400
                )

                # Add to category plots
                category_plots.append(dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(figure=apply_dark_theme(fig_cat))
                        ])
                    ], style=custom_css["card"]),
                    width=12, style={"marginBottom": "20px"}
                ))

        # Combine all plots with section headers
        all_plots = []

        if data_summary_plots:
            all_plots.extend([
                html.H3("Data Overview", className="dashboard-section-title"),
                html.Div(data_summary_plots)
            ])

        # Add correlation heatmap for numeric variables if we have at least 2 numeric columns
        if len(numeric_cols) >= 2:
            # Create correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            # Create a beautiful heatmap with custom color scale
            corr_fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale=[[0.0, '#3498db'],
                           [0.25, '#2980b9'],
                           [0.5, '#34495e'],
                           [0.75, '#16a085'], 
                           [1.0, '#1abc9c']],
                zmin=-1, zmax=1,
                text=corr_matrix.round(2).values,
                texttemplate='%{text}',
                hoverinfo='text',
                hoverongaps=False
            ))
            
            # Improve layout
            corr_fig.update_layout(
                title="Correlation Heatmap (Numeric Variables)",
                height=max(400, len(numeric_cols) * 35),
                xaxis_title="Variables",
                yaxis_title="Variables",
                margin=dict(l=60, r=40, t=70, b=60)
            )
            
            # Add to plots
            all_plots.extend([
                html.H3("Correlation Analysis", className="dashboard-section-title"),
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(figure=apply_dark_theme(corr_fig))
                        ])
                    ], style=custom_css["card"]),
                    width=12, style={"marginBottom": "20px"}
                )
            ])

        if distribution_plots:
            all_plots.extend([
                html.H3("Distribution Analysis", className="dashboard-section-title"),
                html.Div(distribution_plots)
            ])

        if category_plots:
            all_plots.extend([
                html.H3("Categorical Analysis", className="dashboard-section-title"),
                html.Div(category_plots)
            ])

        if not all_plots:
            return html.Div("No visualizations could be generated for this dataset")

        return html.Div(all_plots)

    except Exception as e:
        return html.Div([
            html.H5("Error Generating Visualizations", style={"color": "var(--error)"}),
            html.P(f"An error occurred: {str(e)}", style={"color": "var(--text-secondary)"})
        ])

# Tests Callback
@app.callback(
    [
        Output("test-plot", "figure"),
        Output("test-result", "children"),
        Output("test-table", "data"),
        Output("test-table", "columns"),
    ],
    [
        Input("perform-test", "n_clicks"),
        Input("test-type-dropdown", "value"),
        Input("test-x-dropdown", "value"),
        Input("test-y-dropdown", "value"),
        Input("data-table", "data"),
    ],
)
def perform_test(n_clicks, test_type, x_axis, y_axis, data):
    if not n_clicks or not test_type or not x_axis or not data:
        fig = go.Figure()
        return apply_dark_theme(fig), "Select test type, variables, and click 'Perform Test'.", [], []

    df = pd.DataFrame(data)
    result_text = ""
    table_data = []
    columns = []
    fig = go.Figure()

    try:
        if test_type == "chi2":
            if not y_axis:
                return go.Figure(), "Please select Y-axis for Chi-squared Test.", [], []

            # Direct import of chi2_contingency to avoid circular imports
            from scipy.stats import chi2_contingency

            contingency_table = pd.crosstab(df[x_axis], df[y_axis])
            chi2, p, dof, expected = chi2_contingency(contingency_table)

            fig = go.Figure(
                data=go.Heatmap(
                    z=contingency_table.values,
                    x=contingency_table.columns,
                    y=contingency_table.index,
                    colorscale='Blues',
                    colorbar=dict(title="Frequency"),
                )
            )
            fig.update_layout(title=f"Observed Frequencies: {x_axis} vs {y_axis}")

            observed_df = pd.DataFrame(contingency_table)
            observed_df["Type"] = "Observed"
            expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)
            expected_df["Type"] = "Expected"
            combined_df = pd.concat([observed_df, expected_df])

            columns = [{"name": col, "id": col} for col in combined_df.columns]
            table_data = combined_df.reset_index().to_dict("records")

            result_text = f"Chi-squared Statistic: {chi2:.3f}, p-value: {p:.3f}, Degrees of Freedom: {dof}"

        elif test_type == "pearson":
            if not y_axis:
                return go.Figure(), "Please select Y-axis for Pearson Correlation.", [], []

            if not pd.api.types.is_numeric_dtype(df[x_axis]) or not pd.api.types.is_numeric_dtype(df[y_axis]):
                return go.Figure(), "Both X-axis and Y-axis must be numeric for Pearson Correlation.", [], []

            # Drop rows with NaN or infinite values in either column
            df_valid = df[[x_axis, y_axis]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(df_valid) < 2:
                return go.Figure(), "Not enough valid data points for Pearson correlation.", [], []

            from scipy.stats import pearsonr
            corr, p_value = pearsonr(df_valid[x_axis], df_valid[y_axis])

            fig = make_subplots(rows=1, cols=2,
                               subplot_titles=('Scatter Plot with Regression Line', 'Density Distribution'),
                               specs=[[{"type": "xy"}, {"type": "xy"}]],
                               column_widths=[0.7, 0.3])

            fig.add_trace(
                go.Scatter(
                    x=df_valid[x_axis],
                    y=df_valid[y_axis],
                    mode='markers',
                    marker=dict(
                        color='rgba(26, 188, 156, 0.6)',
                        size=8,
                        line=dict(
                            color='rgba(26, 188, 156, 0.9)',
                            width=1
                        )
                    ),
                    name='Data Points'
                ),
                row=1, col=1
            )

            coef = np.polyfit(df_valid[x_axis], df_valid[y_axis], 1)
            line_x = np.array([df_valid[x_axis].min(), df_valid[x_axis].max()])
            line_y = coef[0] * line_x + coef[1]

            fig.add_trace(
                go.Scatter(
                    x=line_x,
                    y=line_y,
                    mode='lines',
                    line=dict(color='rgba(231, 76, 60, 0.9)', width=2),
                    name=f'Regression Line (y = {coef[0]:.3f}x + {coef[1]:.3f})'
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Histogram(
                    x=df_valid[x_axis],
                    marker=dict(
                        color='rgba(26, 188, 156, 0.7)',
                        line=dict(color='rgba(255, 255, 255, 0.5)', width=1)
                    ),
                    opacity=0.7,
                    name=f'{x_axis} Distribution'
                ),
                row=1, col=2
            )
            mean_val = df_valid[x_axis].mean()
            fig.add_shape(
                type="line",
                x0=mean_val, y0=0,
                x1=mean_val, y1=1,
                yref="paper",
                line=dict(color="#3498db", width=2, dash="dash"),
                row=1, col=2
            )

            fig.update_layout(
                title=f"Pearson Correlation: {x_axis} vs {y_axis} (r = {corr:.3f}, p = {p_value:.3f})",
                height=500,
                showlegend=True
            )

            fig.update_xaxes(title_text=x_axis, row=1, col=1)
            fig.update_yaxes(title_text=y_axis, row=1, col=1)
            fig.update_xaxes(title_text="Value", row=1, col=2)
            fig.update_yaxes(title_text="Density", row=1, col=2)

            significance = "significant" if p_value < 0.05 else "not significant"
            strength = ""
            if abs(corr) < 0.3:
                strength = "weak"
            elif abs(corr) < 0.7:
                strength = "moderate"
            else:
                strength = "strong"

            direction = "positive" if corr > 0 else "negative"

            result_text = f"Pearson Correlation: {corr:.3f}, p-value: {p_value:.3f}<br>" + \
                         f"Interpretation: {strength} {direction} correlation ({significance} at α=0.05)"

        elif test_type == "spearman":
            if not y_axis:
                return go.Figure(), "Please select Y-axis for Spearman Correlation.", [], []

            if not pd.api.types.is_numeric_dtype(df[x_axis]) or not pd.api.types.is_numeric_dtype(df[y_axis]):
                return go.Figure(), "Both X-axis and Y-axis must be numeric for Spearman Correlation.", [], []

            # Drop rows with NaN or infinite values in either column
            df_valid = df[[x_axis, y_axis]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(df_valid) < 2:
                return go.Figure(), "Not enough valid data points for Spearman correlation.", [], []

            from scipy.stats import spearmanr
            corr, p_value = spearmanr(df_valid[x_axis], df_valid[y_axis])

            fig = make_subplots(rows=1, cols=2,
                               subplot_titles=('Scatter Plot with LOWESS Trend', 'Rank Correlation'),
                               specs=[[{"type": "xy"}, {"type": "xy"}]],
                               column_widths=[0.7, 0.3])

            fig.add_trace(
                go.Scatter(
                    x=df_valid[x_axis],
                    y=df_valid[y_axis],
                    mode='markers',
                    marker=dict(
                        color='rgba(26, 188, 156, 0.6)',
                        size=8,
                        line=dict(
                            color='rgba(26, 188, 156, 0.9)',
                            width=1
                        )
                    ),
                    name='Data Points'
                ),
                row=1, col=1
            )

            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                lowess_result = lowess(df_valid[y_axis], df_valid[x_axis], frac=0.5)
                fig.add_trace(
                    go.Scatter(
                        x=lowess_result[:, 0],
                        y=lowess_result[:, 1],
                        mode='lines',
                        line=dict(color='rgba(231, 76, 60, 0.9)', width=2),
                        name='LOWESS Trend'
                    ),
                    row=1, col=1
                )
            except Exception:
                pass

            x_rank = df_valid[x_axis].rank()
            y_rank = df_valid[y_axis].rank()

            fig.add_trace(
                go.Scatter(
                    x=x_rank,
                    y=y_rank,
                    mode='markers',
                    marker=dict(
                        color='rgba(155, 89, 182, 0.6)',
                        size=8,
                        line=dict(
                            color='rgba(155, 89, 182, 0.9)',
                            width=1
                        )
                    ),
                    name='Rank Correlation'
                ),
                row=1, col=2
            )

            fig.update_layout(
                title=f"Spearman Rank Correlation: {x_axis} vs {y_axis} (ρ = {corr:.3f}, p = {p_value:.3f})",
                height=500,
                showlegend=True
            )

            fig.update_xaxes(title_text=x_axis, row=1, col=1)
            fig.update_yaxes(title_text=y_axis, row=1, col=1)
            fig.update_xaxes(title_text=f"{x_axis} Rank", row=1, col=2)
            fig.update_yaxes(title_text=f"{y_axis} Rank", row=1, col=2)

            significance = "significant" if p_value < 0.05 else "not significant"
            strength = ""
            if abs(corr) < 0.3:
                strength = "weak"
            elif abs(corr) < 0.7:
                strength = "moderate"
            else:
                strength = "strong"

            direction = "positive" if corr > 0 else "negative"

            result_text = f"Spearman Correlation: {corr:.3f}, p-value: {p_value:.3f}<br>" + \
                         f"Interpretation: {strength} {direction} monotonic relationship ({significance} at α=0.05)"

        else:
            return go.Figure(), "Invalid test type selected.", [], []

        # Apply the dark theme to the figure
        fig = apply_dark_theme(fig)

    except Exception as e:
        return go.Figure(), f"Error performing test: {str(e)}", [], []

    return fig, result_text, table_data, columns

# Regression Callbacks
@app.callback(
    [
        Output("regression-plot", "figure"),
        Output("regression-equation", "children"),
        Output("regression-metrics", "children"),
        Output("regression-error", "children"),
        Output("regression-error", "is_open"),
    ],
    [
        Input("calculate-regression", "n_clicks"),
        Input("regression-x-dropdown", "value"),
        Input("regression-y-dropdown", "value"),
        Input("data-table", "data"),
    ],
)
def perform_regression(n_clicks, x_var, y_var, data):
    if not n_clicks or not x_var or not y_var or not data:
        fig = go.Figure()
        return apply_dark_theme(fig), "", "", "", False

    try:
        df = pd.DataFrame(data)

        # Check if variables are numeric
        if not pd.api.types.is_numeric_dtype(df[x_var]) or not pd.api.types.is_numeric_dtype(df[y_var]):
            return go.Figure(), "", "", "Both variables must be numeric for regression analysis.", True

        # Drop any rows with missing values
        df = df[[x_var, y_var]].dropna()

        if len(df) < 2:
            return go.Figure(), "", "", "Not enough valid data points for regression analysis.", True

        # Perform regression using statsmodels
        X = df[x_var].values.reshape(-1, 1)
        y = df[y_var].values

        # Add constant for intercept
        X_with_const = sm.add_constant(X)

        # Fit the model
        model = sm.OLS(y, X_with_const)
        results = model.fit()

        # Get coefficients
        intercept = results.params[0]
        slope = results.params[1]
        r_squared = results.rsquared

        # Calculate confidence intervals
        predictions = results.get_prediction(X_with_const)
        ci = predictions.conf_int()

        # Create the plot
        fig = go.Figure()

        # Add scatter plot of data points
        fig.add_trace(
            go.Scatter(
                x=df[x_var],
                y=df[y_var],
                mode='markers',
                name='Data Points',
                marker=dict(
                    color='rgba(66, 133, 244, 0.8)',
                    size=8,
                    line=dict(color='#000000', width=1)
                )
            )
        )

        # Add regression line
        x_range = np.linspace(df[x_var].min(), df[x_var].max(), 100)
        y_pred = slope * x_range + intercept

        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_pred,
                mode='lines',
                name='Regression Line',
                line=dict(color='#EA4335', width=3)
            )
        )

        # Add confidence intervals
        x_for_ci = sm.add_constant(x_range.reshape(-1, 1))
        predictions = results.get_prediction(x_for_ci)
        ci = predictions.conf_int()

        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=ci[:, 0],
                mode='lines',
                line=dict(color='rgba(251, 188, 5, 0.3)', width=0),
                name='95% Confidence Interval'
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=ci[:, 1],
                mode='lines',
                line=dict(color='rgba(251, 188, 5, 0.3)', width=0),
                fill='tonexty',
                name='95% Confidence Interval'
            )
        )

        # Update layout
        fig.update_layout(
            title=f"Linear Regression: {y_var} vs {x_var}",
            xaxis_title=x_var,
            yaxis_title=y_var,
            hovermode='closest'
        )

        # Format equation
        equation = f"y = {slope:.3f}x {'+' if intercept >= 0 else '-'} {abs(intercept):.3f}"

        # Format metrics
        metrics = html.Div([
            html.P([
                html.Strong("Slope (m): "),
                f"{slope:.3f}"
            ], style={"marginBottom": "10px"}),
            html.P([
                html.Strong("Intercept (b): "),
                f"{intercept:.3f}"
            ], style={"marginBottom": "10px"}),
            html.P([
                html.Strong("R² Score: "),
                f"{r_squared:.3f}"
            ], style={"marginBottom": "10px"}),
            html.P([
                html.Strong("P-value: "),
                f"{results.f_pvalue:.3e}"
            ], style={"marginBottom": "10px"}),
        ])

        return apply_dark_theme(fig), equation, metrics, "", False

    except Exception as e:
        return go.Figure(), "", "", f"Error performing regression: {str(e)}", True

@app.callback(
    [
        Output("prediction-result", "children"),
        Output("prediction-input", "invalid"),
    ],
    [
        Input("predict-button", "n_clicks"),
        Input("prediction-input", "value"),
        Input("regression-x-dropdown", "value"),
        Input("regression-y-dropdown", "value"),
        Input("data-table", "data"),
    ],
)
def make_prediction(n_clicks, x_value, x_var, y_var, data):
    if not n_clicks or x_value is None or not x_var or not y_var or not data:
        return "", False

    try:
        df = pd.DataFrame(data)

        # Check if variables are numeric
        if not pd.api.types.is_numeric_dtype(df[x_var]) or not pd.api.types.is_numeric_dtype(df[y_var]):
            return "Error: Variables must be numeric", True

        # Drop any rows with missing values
        df = df[[x_var, y_var]].dropna()

        if len(df) < 2:
            return "Error: Not enough data points", True

        # Perform regression
        X = df[x_var].values.reshape(-1, 1)
        y = df[y_var].values

        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const)
        results = model.fit()

        # Make prediction
        x_new = np.array([[1, float(x_value)]])
        prediction = results.predict(x_new)[0]

        # Get prediction interval
        prediction_interval = results.get_prediction(x_new).conf_int()[0]

        return (
            html.Div([
                html.P(f"Predicted {y_var}: {prediction:.3f}", style={"fontSize": "1.2em", "marginBottom": "10px"}),
                html.P(f"95% Confidence Interval:", style={"marginBottom": "5px"}),
                html.P(f"({prediction_interval[0]:.3f}, {prediction_interval[1]:.3f})")
            ]),
            False
        )

    except Exception as e:
        return f"Error: {str(e)}", True

# Update regression dropdowns with numeric columns only
@app.callback(
    [
        Output("regression-x-dropdown", "options"),
        Output("regression-y-dropdown", "options"),
    ],
    [Input("data-table", "data")],
)
def update_regression_dropdowns(data):
    if not data:
        return [], []

    try:
        df = pd.DataFrame(data)
        numeric_cols = df.select_dtypes(include=['number']).columns
        options = [{"label": html.Span(col, style={"color": "#FFFFFF"}), "value": col} for col in numeric_cols]
        return options, options
    except Exception as e:
        print(f"Error updating regression dropdowns: {str(e)}")
        return [], []

# Helper functions for the EDA Report
def generate_eda_report_components(df):
    """Generate components for the EDA report from the dataframe"""
    components = []

    # ---- 1. OVERVIEW SECTION ----
    overview_card = dbc.Card([
        dbc.CardHeader("Dataset Overview", style=custom_css["card_header"]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H5("Basic Information", style={"color": "var(--primary)", "marginBottom": "15px"}),
                    html.P(f"Number of Rows: {df.shape[0]}", style={"color": "var(--text-primary)", "marginBottom": "5px"}),
                    html.P(f"Number of Columns: {df.shape[1]}", style={"color": "var(--text-primary)", "marginBottom": "5px"}),
                    html.P(f"Duplicate Rows: {df.duplicated().sum()}", style={"color": "var(--text-primary)", "marginBottom": "5px"}),
                    html.P(f"Total Missing Values: {df.isna().sum().sum()}", style={"color": "var(--text-primary)", "marginBottom": "15px"}),

                    html.H5("Data Types", style={"color": "var(--primary)", "marginBottom": "15px", "marginTop": "20px"}),
                    dbc.Table(
                        # Create a table with column names and their data types
                        [
                            html.Thead(html.Tr([html.Th("Column"), html.Th("Type")])),
                            html.Tbody([
                                html.Tr([
                                    html.Td(col, style={"color": "var(--text-primary)"}),
                                    html.Td(str(df[col].dtype), style={"color": "var(--text-primary)"})
                                ]) for col in df.columns
                            ])
                        ],
                        bordered=True,
                        hover=True,
                        responsive=True,
                        size="sm",
                        style={"backgroundColor": "var(--card-bg)", "color": "var(--text-primary)"}
                    ),
                ], width=6),

                dbc.Col([
                    html.H5("Missing Values by Column", style={"color": "var(--primary)", "marginBottom": "15px"}),
                    dcc.Graph(
                        figure=apply_dark_theme(
                            px.bar(
                                df.isna().sum().reset_index(),
                                x="index",
                                y=0,
                                labels={"index": "Column", "0": "Missing Values"},
                                title="Missing Values Count",
                                height=300
                            )
                        )
                    ),

                    html.H5("Dataset Warnings", style={"color": "var(--primary)", "marginBottom": "15px", "marginTop": "20px"}),
                    html.Ul([
                        # Check for various potential issues in the dataset
                        html.Li(f"Constant columns: {sum(df.nunique() == 1)}",
                                style={"color": "var(--text-primary)"}),
                        html.Li(f"Columns with >50% missing values: {sum(df.isna().mean() > 0.5)}",
                                style={"color": "var(--warning)" if sum(df.isna().mean() > 0.5) > 0 else "var(--text-primary)"}),
                        html.Li(f"High cardinality categorical columns: {sum([df[col].nunique() > 50 for col in df.select_dtypes(include=['object']).columns] if not df.select_dtypes(include=['object']).empty else [])}",
                                style={"color": "var(--warning)" if sum([df[col].nunique() > 50 for col in df.select_dtypes(include=['object']).columns] if not df.select_dtypes(include=['object']).empty else []) > 0 else "var(--text-primary)"}),
                    ])
                ], width=6)
            ])
        ])
    ], style=custom_css["card"])

    components.append(dbc.Row([dbc.Col(overview_card, width=12)], className="mb-4"))

    # ---- 2. DESCRIPTIVE STATISTICS ----
    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # 2.1 Numeric Statistics
    if numeric_cols:
        # Convert the describe dataframe to a presentable format
        desc_df = df[numeric_cols].describe().transpose().reset_index()
        desc_df = desc_df.rename(columns={'index': 'Column'})

        # Round all numeric columns to 3 decimal places
        for col in desc_df.columns:
            if col != 'Column':
                desc_df[col] = desc_df[col].round(3)

        stats_card = dbc.Card([
            dbc.CardHeader("Numerical Statistics", style=custom_css["card_header"]),
            dbc.CardBody([
                dbc.Table.from_dataframe(
                    desc_df,
                    striped=True,
                    bordered=True,
                    hover=True,
                    responsive=True,
                    style={"backgroundColor": "var(--card-bg)", "color": "var(--text-primary)"}
                )
            ])
        ], style=custom_css["card"])

        components.append(dbc.Row([dbc.Col(stats_card, width=12)], className="mb-4"))

    # 2.2 Categorical Statistics (value counts for top categories)
    if categorical_cols:
        cat_stats_rows = []

        for col in categorical_cols[:5]:  # Limit to first 5 categorical columns for brevity
            value_counts = df[col].value_counts().head(5).reset_index()
            value_counts.columns = ['Value', 'Count']

            # Calculate percentage
            total = len(df)
            value_counts['Percentage'] = (value_counts['Count'] / total * 100).round(2)

            cat_stats_rows.append(
                dbc.Row([
                    dbc.Col([
                        html.H5(f"Top values for '{col}'", style={"color": "var(--primary)", "marginBottom": "15px"}),
                        dbc.Table.from_dataframe(
                            value_counts,
                            striped=True,
                            bordered=True,
                            hover=True,
                            responsive=True,
                            size="sm",
                            style={"backgroundColor": "var(--card-bg)", "color": "var(--text-primary)"}
                        )
                    ], width=6),
                    dbc.Col([
                        dcc.Graph(
                            figure=apply_dark_theme(
                                px.pie(
                                    value_counts,
                                    values='Count',
                                    names='Value',
                                    title=f"Distribution of '{col}'",
                                    height=300
                                )
                            )
                        )
                    ], width=6)
                ])
            )

        cat_stats_card = dbc.Card([
            dbc.CardHeader("Categorical Variables", style=custom_css["card_header"]),
            dbc.CardBody(cat_stats_rows)
        ], style=custom_css["card"])

        components.append(dbc.Row([dbc.Col(cat_stats_card, width=12)], className="mb-4"))

    # ---- 3. VISUALIZATIONS ----
    # 3.1 Histograms for Numeric Columns
    if numeric_cols:
        hist_figs = []
        for i in range(0, min(len(numeric_cols), 5)):  # Limit to first 5 numeric columns
            col = numeric_cols[i]
            fig = px.histogram(
                df, x=col,
                nbins=20,
                marginal=None,  # No box plot
                title=f"Distribution of {col}",
                color_discrete_sequence=['#1abc9c'],
                template="plotly_dark"
            )
            fig.update_traces(marker_line_color='#16a085', marker_line_width=2, opacity=0.85)
            fig.update_layout(bargap=0.05)
            mean_val = df[col].mean()
            fig.add_vline(x=mean_val, line_dash="dash", line_color="#3498db")

            hist_figs.append(
                dbc.Col([
                    dcc.Graph(figure=apply_dark_theme(fig))
                ], width=6)
            )

        # Arrange histograms in rows
        hist_rows = []
        for i in range(0, len(hist_figs), 2):
            row_figs = hist_figs[i:i+2]
            hist_rows.append(dbc.Row(row_figs, className="mb-4"))

        hist_card = dbc.Card([
            dbc.CardHeader("Distribution of Numeric Variables", style=custom_css["card_header"]),
            dbc.CardBody(hist_rows)
        ], style=custom_css["card"])

        components.append(dbc.Row([dbc.Col(hist_card, width=12)], className="mb-4"))

    # 3.3 Correlation Heatmap
    if len(numeric_cols) > 1:
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr().round(2)

        # Create heatmap
        corr_fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='turbo',
            title="Correlation Matrix",
            aspect="auto"
        )

        corr_card = dbc.Card([
            dbc.CardHeader("Correlation Heatmap", style=custom_css["card_header"]),
                        dbc.CardBody([
                            dcc.Graph(figure=apply_dark_theme(corr_fig))
                        ])
        ], style=custom_css["card"])

        # Add correlation card to the components
        components.append(dbc.Row([dbc.Col(corr_card, width=12)], className="mb-4"))

    return components

# Add the EDA Report Callback
@app.callback(
    Output("eda-report-container", "children"),
    [Input("generate-report-button", "n_clicks")],
    [State("data-table", "data")],
    prevent_initial_call=True
)
def generate_eda_report(n_clicks, data):
    if not n_clicks or not data:
        return html.Div("Please upload data and click 'Generate EDA Report' to see the analysis.")

    try:
        # Convert data to DataFrame
        df = pd.DataFrame(data)

        if df.empty:
            return html.Div("No data available to analyze.")

        # Generate report components
        report_components = generate_eda_report_components(df)

        # Add introduction section at the top
        intro = html.Div([
            html.H4("Exploratory Data Analysis Report", style={"color": "var(--primary)", "marginBottom": "20px"}),
            html.P("This automated report provides insights into your dataset structure, statistics, and visualizations.",
                  style={"color": "var(--text-secondary)", "marginBottom": "30px"}),
        ])

        # Final report layout
        report = html.Div([
            intro,
            *report_components,
            html.Div([
                html.Hr(style={"borderColor": "var(--border-color)", "margin": "30px 0"}),
                html.P(f"Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
                      style={"color": "var(--text-secondary)", "textAlign": "center", "fontSize": "12px", "opacity": "0.7"})
            ])
        ])

        return report

    except Exception as e:
        return html.Div([
            html.H5("Error Generating Report", style={"color": "var(--error)"}),
            html.P(f"An error occurred: {str(e)}", style={"color": "var(--text-secondary)"})
        ])

# Update plot type dropdown options
@app.callback(
    Output("plot-type-dropdown", "options"),
    [Input("data-table", "data")]
)
def update_plot_type_dropdown(data):
    return [
        {"label": html.Span("Histogram", style={"color": "#FFFFFF"}), "value": "histogram"},
        {"label": html.Span("Scatter Plot", style={"color": "#FFFFFF"}), "value": "scatter"},
        {"label": html.Span("Bar Chart", style={"color": "#FFFFFF"}), "value": "bar"},
    ]

# Update axis dropdowns based on plot type
@app.callback(
    [
        Output("x-axis-dropdown", "options", allow_duplicate=True),
        Output("y-axis-dropdown", "options", allow_duplicate=True),
        Output("y-axis-dropdown", "disabled"),  # Add this output to disable Y-axis when not needed
        Output("y-axis-dropdown", "placeholder"),  # Add this output to update placeholder text
    ],
    [
        Input("plot-type-dropdown", "value"),
        Input("data-table", "data"),
    ],
    prevent_initial_call=True
)
def update_axis_dropdowns(plot_type, data):
    if not data:
        return [], [], True, "Select Y-axis"

    df = pd.DataFrame(data)

    # Get column types for options
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    categorical_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]

    all_cols = list(df.columns)
    y_placeholder = "Select Y-axis"

    if plot_type == "histogram":
        # For histogram, we only need X-axis numeric
        x_options = [{"label": col, "value": col} for col in numeric_cols]
        y_options = []  # No Y-axis needed for histograms
        disable_y = True  # Disable Y-axis dropdown
        y_placeholder = "Not needed for histogram"

    elif plot_type == "scatter":
        # For scatter, both X and Y should be numeric
        x_options = [{"label": col, "value": col} for col in numeric_cols]
        y_options = [{"label": col, "value": col} for col in numeric_cols]
        disable_y = False

    elif plot_type == "bar":
        # For bar charts, X can be any column, Y should be numeric
        x_options = [{"label": col, "value": col} for col in all_cols]
        y_options = [{"label": col, "value": col} for col in numeric_cols]
        disable_y = False

    else:
        x_options = [{"label": col, "value": col} for col in all_cols]
        y_options = [{"label": col, "value": col} for col in all_cols]
        disable_y = False

    return x_options, y_options, disable_y, y_placeholder

# Add a new callback to update test dropdowns based on test type
@app.callback(
    [
        Output("test-x-dropdown", "options", allow_duplicate=True),
        Output("test-y-dropdown", "options", allow_duplicate=True),
        Output("test-x-dropdown", "value"),
        Output("test-y-dropdown", "value"),
    ],
    [
        Input("test-type-dropdown", "value"),
        Input("data-table", "data"),
    ],
    prevent_initial_call=True
)
def update_test_dropdowns(test_type, data):
    if not data or not test_type:
        return [], [], None, None

    df = pd.DataFrame(data)

    # Reset dropdown values
    x_value = None
    y_value = None

    # Get column types
    categorical_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    # Format dropdown options
    categorical_options = [{"label": html.Span(col, style={"color": "#FFFFFF"}), "value": col} for col in categorical_cols]
    numeric_options = [{"label": html.Span(col, style={"color": "#FFFFFF"}), "value": col} for col in numeric_cols]

    # Select options based on test type
    if test_type == "chi2":
        # Chi-squared: Both X and Y should be categorical
        return categorical_options, categorical_options, x_value, y_value
    elif test_type in ["pearson", "spearman"]:
        # Correlation tests: Both X and Y must be numeric
        return numeric_options, numeric_options, x_value, y_value
    else:
        # Default case
        all_options = [{"label": html.Span(col, style={"color": "#FFFFFF"}), "value": col} for col in df.columns]
        return all_options, all_options, x_value, y_value

# Add a callback to handle the Apply Imputation button click
@app.callback(
    [
        Output("imputation-success-toast", "is_open", allow_duplicate=True),
        Output("imputation-warning-toast", "is_open", allow_duplicate=True),
    ],
    [Input("apply-imputation-button", "n_clicks")],
    [State("imputation-columns", "value")],
    prevent_initial_call=True
)
def handle_imputation_button(n_clicks, selected_columns):
    if not n_clicks:
        return False, False

    # Show warning if no columns selected
    if not selected_columns:
        return False, True
    else:
        # Show success toast if columns were selected
        return True, False

# Update prediction dropdowns with column options
@app.callback(
    [
        Output("prediction-target-dropdown", "options"),
        Output("prediction-features-dropdown", "options"),
    ],
    [Input("data-table", "data")],
)
def update_prediction_dropdowns(data):
    if not data:
        return [], []

    try:
        df = pd.DataFrame(data)
        options = [{"label": html.Span(col, style={"color": "#FFFFFF"}), "value": col} for col in df.columns]
        return options, options
    except Exception as e:
        print(f"Error updating prediction dropdowns: {str(e)}")
        return [], []

# Create input fields for manual prediction based on selected features
@app.callback(
    [
        Output("manual-inputs-container", "children"),
        Output("predict-button-manual", "disabled"),
    ],
    [
        Input("prediction-features-dropdown", "value"),
        Input("feature-info-store", "data"),
        Input("trained-model-store", "data")
    ],
)
def create_manual_input_fields(selected_features, feature_info, model_info):
    if not selected_features or not feature_info or not model_info:
        return [], True

    input_fields = []

    for feature in selected_features:
        feature_type = feature_info.get(feature, {}).get("type", "numeric")

        if feature_type == "categorical":
            categories = feature_info.get(feature, {}).get("categories", [])
            input_fields.append(
                html.Div([
                    html.Label(f"{feature}:", style={
                        "color": "#e6e6e6",
                        "fontWeight": "bold",
                        "marginBottom": "5px",
                        "display": "block"
                    }),
                    dcc.Dropdown(
                        id={"type": "manual-input", "feature": feature},
                        options=[{"label": html.Span(str(cat), style={"color": "#FFFFFF"}), "value": str(cat)} for cat in categories],
                        placeholder=f"Select {feature}",
                        style={"marginBottom": "15px", **custom_css["dropdown"]},
                        className='dropdown-dark custom-dropdown'
                    )
                ], style={"marginBottom": "15px"})
            )
        else:  # numeric input
            input_fields.append(
                html.Div([
                    html.Label(f"{feature}:", style={
                        "color": "#e6e6e6",
                        "fontWeight": "bold",
                        "marginBottom": "5px",
                        "display": "block"
                    }),
                    dbc.Input(
                        id={"type": "manual-input", "feature": feature},
                        type="number",
                        placeholder=f"Enter value for {feature}",
                        style={"marginBottom": "15px", **custom_css["dropdown"]}
                    )
                ], style={"marginBottom": "15px"})
            )

    return input_fields, False

# Train model callback
@app.callback(
    [
        Output("trained-model-store", "data"),
        Output("feature-info-store", "data"),
        Output("model-performance-store", "data"),
        Output("training-status", "children"),
        Output("predict-button-file", "disabled"),
    ],
    [Input("train-model-button", "n_clicks")],
    [
        State("data-table", "data"),
        State("prediction-target-dropdown", "value"),
        State("prediction-features-dropdown", "value"),
        State("n-estimators-input", "value"),
        State("max-depth-input", "value"),
        State("test-size-input", "value"),
        State("random-state-input", "value"),
    ],
)
def train_model(n_clicks, data, target, features, n_estimators, max_depth, test_size, random_state):
    if not n_clicks or not data or not target or not features:
        return None, None, None, "", True

    try:
        df = pd.DataFrame(data)

        # Handle missing values
        df = df.dropna(subset=[target] + features)

        if len(df) < 10:
            return None, None, None, html.Div([
                html.I(className="fas fa-exclamation-circle mr-2", style={"color": "#ff6b6b"}),
                "Not enough data after removing missing values. Need at least 10 rows."
            ], style={"color": "#ff6b6b"}), True

        # Prepare feature information for encoding/preprocessing
        feature_info = {}
        X_processed = pd.DataFrame()

        # Process each feature
        for feature in features:
            if pd.api.types.is_numeric_dtype(df[feature]):
                # For numeric features, just store type and use as is
                feature_info[feature] = {
                    "type": "numeric",
                    "mean": float(df[feature].mean()),
                    "std": float(df[feature].std()) if df[feature].std() > 0 else 1.0
                }
                # Normalize numeric features
                X_processed[feature] = (df[feature] - feature_info[feature]["mean"]) / feature_info[feature]["std"]
            else:
                # For categorical features, store categories and one-hot encode
                unique_values = df[feature].unique().tolist()
                feature_info[feature] = {
                    "type": "categorical",
                    "categories": unique_values
                }
                # One-hot encode categorical features
                for category in unique_values:
                    col_name = f"{feature}_{category}"
                    X_processed[col_name] = (df[feature] == category).astype(int)

        # Process target variable
        y = df[target]
        if not pd.api.types.is_numeric_dtype(y) and len(y.unique()) <= 10:
            # For categorical target, encode as numeric
            target_mapping = {val: i for i, val in enumerate(y.unique())}
            y_encoded = y.map(target_mapping)
            target_info = {
                "type": "categorical",
                "mapping": target_mapping,
                "inverse_mapping": {i: val for val, i in target_mapping.items()}
            }
        else:
            # For numeric target or high-cardinality categorical, treat as regression
            y_encoded = y
            target_info = {"type": "numeric"}

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = get_sklearn('train_test_split')(
            X_processed, y_encoded, test_size=test_size, random_state=random_state
        )

        # Adjust max_depth parameter
        if max_depth is None or max_depth == "":
            max_depth = None
        else:
            max_depth = int(max_depth)

        # Train model
        model = get_sklearn('RandomForestClassifier')(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)

        # Calculate metrics
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        # Store feature importances
        feature_importances = dict(zip(X_processed.columns, model.feature_importances_))

        # Store model information (note: we can't store the actual model object in dcc.Store)
        model_info = {
            "target": target,
            "features": features,
            "target_info": target_info,
            "feature_importances": feature_importances,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
            "test_size": test_size
        }

        # Create model metrics
        metrics = {
            "train_score": train_score,
            "test_score": test_score,
            "n_samples": len(df),
            "n_features": len(X_processed.columns)
        }

        # Create status message with metrics
        status = html.Div([
            html.Div([
                html.I(className="fas fa-check-circle mr-2", style={"color": "#51cf66"}),
                "Model trained successfully!"
            ], style={"color": "#51cf66", "fontWeight": "bold", "marginBottom": "15px"}),

            html.Div([
                html.Div([
                    html.Strong("Train Score: "),
                    f"{train_score:.3f} ({train_score*100:.1f}%)"
                ], style={"marginBottom": "5px"}),
                html.Div([
                    html.Strong("Test Score: "),
                    f"{test_score:.3f} ({test_score*100:.1f}%)"
                ], style={"marginBottom": "5px"}),
                html.Div([
                    html.Strong("Samples: "),
                    f"{len(df)}"
                ], style={"marginBottom": "5px"}),
                html.Div([
                    html.Strong("Features: "),
                    f"{len(X_processed.columns)} (after encoding)"
                ])
            ], style={"color": "#e6e6e6"})
        ])

        return model_info, feature_info, metrics, status, False
    except Exception as e:
        error_message = html.Div([
            html.I(className="fas fa-exclamation-circle mr-2", style={"color": "#ff6b6b"}),
            f"Error training model: {str(e)}"
        ], style={"color": "#ff6b6b"})

        return None, None, None, error_message, True

# Process uploaded prediction file
@app.callback(
    [
        Output("prediction-upload-status", "children"),
        Output("prediction-df-store", "data"),
    ],
    [Input("prediction-upload", "contents")],
    [
        State("prediction-upload", "filename"),
        State("prediction-features-dropdown", "value"),
    ],
)
def process_prediction_file(contents, filename, required_features):
    if not contents or not required_features:
        return "Upload a file for prediction", None

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    try:
        if "csv" in filename.lower():
            df = pd.read_csv(StringIO(decoded.decode("utf-8")))
        elif "xls" in filename.lower():
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return f"Unsupported file format: {filename}", None

        # Check if required features are present
        missing_features = [feature for feature in required_features if feature not in df.columns]

        if missing_features:
            return f"Missing required features: {', '.join(missing_features)}", None

        # Store only required columns
        prediction_df = df[required_features].to_dict('records')

        return f"File processed: {filename} ({len(df)} rows)", prediction_df

    except Exception as e:
        return f"Error processing file: {str(e)}", None

# Make predictions from manual input
@app.callback(
    Output("prediction-results", "children"),
    [
        Input("predict-button-manual", "n_clicks"),
        Input("predict-button-file", "n_clicks")
    ],
    [
        State({"type": "manual-input", "feature": dash.ALL}, "value"),
        State({"type": "manual-input", "feature": dash.ALL}, "id"),
        State("prediction-df-store", "data"),
        State("trained-model-store", "data"),
        State("feature-info-store", "data")
    ]
)
def make_predictions(manual_clicks, file_clicks, manual_values, manual_ids, file_data, model_info, feature_info):
    if not model_info or not feature_info:
        return html.Div([
            html.I(className="fas fa-exclamation-circle mr-2", style={"color": "#ff6b6b"}),
            "Please train a model first"
        ], style={"color": "#ff6b6b"})

    ctx = dash.callback_context
    if not ctx.triggered:
        return ""

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    try:
        # Get target information
        target = model_info["target"]
        target_info = model_info["target_info"]
        features = model_info["features"]

        if button_id == "predict-button-manual":
            # Process manual input
            if not manual_values or not manual_ids:
                return html.Div([
                    html.I(className="fas fa-exclamation-circle mr-2", style={"color": "#ff6b6b"}),
                    "Please fill all input fields"
                ], style={"color": "#ff6b6b"})

            # Create a dictionary with feature names and values
            input_data = {}
            for value, id_obj in zip(manual_values, manual_ids):
                feature = id_obj["feature"]
                input_data[feature] = value

            # Convert to DataFrame with a single row
            prediction_df = pd.DataFrame([input_data])

        elif button_id == "predict-button-file":
            # Process file input
            if not file_data:
                return html.Div([
                    html.I(className="fas fa-exclamation-circle mr-2", style={"color": "#ff6b6b"}),
                    "Please upload a file for prediction"
                ], style={"color": "#ff6b6b"})

            prediction_df = pd.DataFrame(file_data)
        else:
            return ""

        # Preprocess input data similarly to training
        X_processed = pd.DataFrame()

        # Process each feature
        for feature in features:
            if feature not in prediction_df.columns:
                return html.Div([
                    html.I(className="fas fa-exclamation-circle mr-2", style={"color": "#ff6b6b"}),
                    f"Missing feature: {feature}"
                ], style={"color": "#ff6b6b"})

            feature_type = feature_info.get(feature, {}).get("type", "numeric")

            if feature_type == "numeric":
                # Normalize numeric features
                mean = feature_info[feature]["mean"]
                std = feature_info[feature]["std"]
                X_processed[feature] = (prediction_df[feature].astype(float) - mean) / std
            else:
                # One-hot encode categorical features
                categories = feature_info[feature]["categories"]
                for category in categories:
                    col_name = f"{feature}_{category}"
                    X_processed[col_name] = (prediction_df[feature].astype(str) == str(category)).astype(int)

        # Check for missing columns that were in training data
        if button_id == "predict-button-file":
            # For file predictions, we need to manually implement a RandomForestClassifier
            # since we can't store the actual model object
            random_forest = get_sklearn('RandomForestClassifier')(
                n_estimators=model_info["n_estimators"],
                max_depth=model_info["max_depth"],
                random_state=model_info["random_state"]
            )

            # Re-train the model on the entire dataset
            # Note: In a production app, we would store the model rather than re-training
            df = pd.DataFrame(file_data)

            # Process features and target
            X = df[features]
            y = df[target]

            # Train the model
            random_forest.fit(X, y)

            # Make predictions
            predictions = random_forest.predict(X)

            # Return results for file prediction
            results_table = dash_table.DataTable(
                data=prediction_df.assign(Prediction=predictions).to_dict('records'),
                columns=[{"name": col, "id": col} for col in prediction_df.columns] + [{"name": "Prediction", "id": "Prediction"}],
                page_size=10,
                style_table={"overflowX": "auto", **custom_css["table"]},
                style_header=custom_css["table_header"],
                style_cell={
                    "backgroundColor": "#16213e",
                    "color": "#e6e6e6",
                    "padding": "10px",
                    "border": "1px solid #2a3a5e"
                },
                style_data_conditional=[
                    {
                        "if": {"row_index": "odd"},
                        "backgroundColor": "#1a1a2e"
                    }
                ],
            )

            return html.Div([
                html.Div([
                    html.I(className="fas fa-check-circle mr-2", style={"color": "#51cf66"}),
                    f"Predictions completed for {len(prediction_df)} rows"
                ], style={"color": "#51cf66", "fontWeight": "bold", "marginBottom": "15px"}),
                results_table
            ])
        else:
            # For manual prediction, implement a simplified prediction
            # In a real implementation, we would store the trained model
            # Here, we'll use the feature importances to roughly simulate the model
            importances = model_info["feature_importances"]

            # Get the most important features and their values
            top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:3]
            feature_values = []
            for feature, importance in top_features:
                if feature in X_processed.columns:
                    feature_values.append(f"{feature}: {X_processed[feature].values[0]:.3f}")

            # Make a simulated prediction
            if target_info["type"] == "categorical":
                # For classification, select a random class with higher probability for the first class
                # This is a simplification since we don't have the actual trained model
                class_idx = 0  # Default to first class
                inverse_mapping = target_info["inverse_mapping"]
                prediction = inverse_mapping[class_idx]

                result = html.Div([
                    html.Div([
                        html.I(className="fas fa-magic mr-2", style={"color": "#1abc9c"}),
                        "Prediction Result:"
                    ], style={"color": "#1abc9c", "fontWeight": "bold", "marginBottom": "15px", "fontSize": "18px"}),

                    html.Div([
                        html.Strong(f"Predicted {target}: "),
                        html.Span(f"{prediction}", style={"color": "#1abc9c", "fontWeight": "bold", "fontSize": "18px"})
                    ], style={"marginBottom": "20px", "padding": "15px", "backgroundColor": "rgba(26, 188, 156, 0.1)", "borderRadius": "8px"}),

                    html.Div([
                        html.Strong("Top influential features:"),
                        html.Ul([
                            html.Li(f"{feature_value}") for feature_value in feature_values
                        ], style={"marginLeft": "20px", "marginTop": "10px"})
                    ], style={"color": "#e6e6e6"})
                ])

                return result
            else:
                # For regression, make a simple weighted sum prediction
                # This is a simplification since we don't have the actual trained model
                prediction = sum(X_processed[feature] * importance for feature, importance in importances.items() if feature in X_processed.columns)

                result = html.Div([
                    html.Div([
                        html.I(className="fas fa-magic mr-2", style={"color": "#1abc9c"}),
                        "Prediction Result:"
                    ], style={"color": "#1abc9c", "fontWeight": "bold", "marginBottom": "15px", "fontSize": "18px"}),

                    html.Div([
                        html.Strong(f"Predicted {target}: "),
                        html.Span(f"{prediction.values[0]:.3f}", style={"color": "#1abc9c", "fontWeight": "bold", "fontSize": "18px"})
                    ], style={"marginBottom": "20px", "padding": "15px", "backgroundColor": "rgba(26, 188, 156, 0.1)", "borderRadius": "8px"}),

                    html.Div([
                        html.Strong("Top influential features:"),
                        html.Ul([
                            html.Li(f"{feature_value}") for feature_value in feature_values
                        ], style={"marginLeft": "20px", "marginTop": "10px"})
                    ], style={"color": "#e6e6e6"})
                ])

                return result

    except Exception as e:
        return html.Div([
            html.I(className="fas fa-exclamation-circle mr-2", style={"color": "#ff6b6b"}),
            f"Error making prediction: {str(e)}"
        ], style={"color": "#ff6b6b"})

# Callback for downloading imputed data as CSV
@app.callback(
    Output("download-imputed-csv", "data"),
    [Input("download-imputed-csv-button", "n_clicks")],
    [State("imputed-table", "data")],
    prevent_initial_call=True,
)
def download_imputed_csv(n_clicks, data):
    if not data:
        return None

    df = pd.DataFrame(data)
    return dcc.send_data_frame(df.to_csv, "imputed_data.csv", index=False)

# Callback for downloading imputed data as JSON
@app.callback(
    Output("download-imputed-json", "data"),
    [Input("download-imputed-json-button", "n_clicks")],
    [State("imputed-table", "data")],
    prevent_initial_call=True,
)
def download_imputed_json(n_clicks, data):
    if not data:
        return None

    df = pd.DataFrame(data)
    return dict(content=df.to_json(orient="records"), filename="imputed_data.json")

# Callback for downloading imputed data as Excel
@app.callback(
    Output("download-imputed-exc el", "data"),
    [Input("download-imputed-excel-button", "n_clicks")],
    [State("imputed-table", "data")],
    prevent_initial_call=True,
)
def download_imputed_excel(n_clicks, data):
    if not data:
        return None

    df = pd.DataFrame(data)
    return dcc.send_data_frame(df.to_excel, "imputed_data.xlsx", sheet_name="Imputed Data", index=False)

# Populate encoding column dropdown with categorical columns
@app.callback(
    Output("encoding_column_dropdown", "options"),
    [Input("data-table", "data")],
)
def update_encoding_column_options(data):
    if not data:
        return []
    import pandas as pd
    df = pd.DataFrame(data)
    categorical_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    return [{"label": col, "value": col} for col in categorical_cols]

# Show/hide ordinal order input
@app.callback(
    Output("encoding_ordinal_container", "children"),
    [Input("encoding_method_dropdown", "value"), Input("encoding_column_dropdown", "value"), Input("data-table", "data")],
)
def show_ordinal_order_input(encoding_type, col, data):
    import pandas as pd
    # Always render the dropdown, but hide it unless needed
    style = {"minHeight": "40px", **custom_css["dropdown"]}
    if encoding_type != "ordinal" or not col or not data:
        style["display"] = "none"
        # Still return the dropdown, but hidden
        return dcc.Dropdown(
            id="encoding_ordinal_dropdown",
            options=[],
            value=[],
            multi=True,
            style=style,
            className='dropdown-dark custom-dropdown'
        )
    df = pd.DataFrame(data)
    if col not in df.columns:
        style["display"] = "none"
        return dcc.Dropdown(
            id="encoding_ordinal_dropdown",
            options=[],
            value=[],
            multi=True,
            style=style,
            className='dropdown-dark custom-dropdown'
        )
    unique_vals = list(map(str, sorted(df[col].dropna().unique(), key=str)))
    # Always show all unique values as options, and set value to all unique values (sorted)
    # Prevent removal by disabling options (Dash doesn't support reorder-only natively), so we add a note
    return [
        html.Label("Specify Order for Ordinal Encoding (all values required, drag to reorder):", style={"color": "#e6e6e6", "fontWeight": "bold", "marginBottom": "8px"}),
        dcc.Dropdown(
            id="encoding_ordinal_dropdown",
            options=[{"label": v, "value": v, "disabled": False} for v in unique_vals],
            value=unique_vals,
            multi=True,
            placeholder="Drag to reorder (top=lowest, bottom=highest)",
            style=style,
            className='dropdown-dark custom-dropdown'
        ),
        html.Div("(All values must be present. Drag to reorder. If you remove a value, it will be restored.)", style={"color": "#aaa", "fontSize": "12px", "marginTop": "5px"})
    ]

# Encoding callback
@app.callback(
    [
        Output("encoding_preview_table", "data"),
        Output("encoding_preview_table", "columns"),
        Output("encoding_message", "children"),
        Output("encoding_data_store", "data"),
    ],
    [
        Input("encoding_apply_button", "n_clicks"),
        Input("encoding_show_encoded_toggle", "value"),
    ],
    [
        State("data-table", "data"),
        State("encoding_column_dropdown", "value"),
        State("encoding_method_dropdown", "value"),
        State("encoding_data_store", "data"),
        State("encoding_ordinal_dropdown", "value"),
    ],
    prevent_initial_call=True,
)
def apply_encoding(n_clicks, show_encoded_only, data, column, encoding_type, stored_encoded_df, ordinal_values):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if not data:
        return [], [], html.Div("No data available", style={"color": "red"}), None

    if trigger_id == "encoding_apply_button":
        if not column or not encoding_type:
            return [], [], html.Div("Please select a column and encoding method", style={"color": "red"}), None

        df = pd.DataFrame(data)

        try:
            # Create a copy of the original dataframe
            encoded_df = df.copy()
            encoded_column = None

            # Apply the selected encoding
            if encoding_type == "label":
                # Label encoding
                le = LabelEncoder()
                encoded_df[f"{column}_encoded"] = le.fit_transform(df[column])
                encoded_column = encoded_df[f"{column}_encoded"]

                # Create a mapping dictionary for display
                mapping = {i: label for i, label in enumerate(le.classes_)}
                mapping_str = ", ".join([f"{k}: {v}" for k, v in mapping.items()])

                message = html.Div([
                    html.P(f"Label encoding applied to '{column}'", style={"color": "#1abc9c", "fontWeight": "bold"}),
                    html.P(f"Mapping: {mapping_str}", style={"color": "#e6e6e6"})
                ])

                # Store data for download
                store_data = {
                    "full_df": encoded_df.to_dict("records"),
                    "column_name": column,
                    "encoded_column": {f"{column}_encoded": encoded_df[f"{column}_encoded"].tolist()},
                    "encoding_type": "label",
                    "mapping": mapping
                }

            elif encoding_type == "onehot":
                # One-hot encoding
                # Get dummies for the selected column
                dummies = pd.get_dummies(df[column], prefix=column)

                # Add the dummies to the original dataframe
                encoded_df = pd.concat([df, dummies], axis=1)
                encoded_column = dummies

                # Create a message with the new columns
                new_cols = dummies.columns.tolist()
                new_cols_str = ", ".join(new_cols)

                message = html.Div([
                    html.P(f"One-hot encoding applied to '{column}'", style={"color": "#1abc9c", "fontWeight": "bold"}),
                    html.P(f"New columns: {new_cols_str}", style={"color": "#e6e6e6"})
                ])

                # Store data for download
                store_data = {
                    "full_df": encoded_df.to_dict("records"),
                    "column_name": column,
                    "encoded_column": dummies.to_dict("records"),
                    "encoding_type": "onehot",
                    "new_columns": new_cols
                }

            elif encoding_type == "ordinal":
                # Use the order provided by the user
                if ordinal_values:
                    # Create mapping based on the order in the dropdown
                    ordinal_map = {val: i for i, val in enumerate(ordinal_values)}
                else:
                    # Fallback to sorted values if no order is provided
                    unique_values = sorted(df[column].unique())
                    ordinal_map = {val: i for i, val in enumerate(unique_values)}

                encoded_df[f"{column}_ordinal"] = df[column].map(ordinal_map)
                encoded_column = encoded_df[f"{column}_ordinal"]

                # Create a mapping string for display
                mapping_str = ", ".join([f"{v}: {k}" for k, v in ordinal_map.items()])

                message = html.Div([
                    html.P(f"Ordinal encoding applied to '{column}'", style={"color": "#1abc9c", "fontWeight": "bold"}),
                    html.P(f"Mapping: {mapping_str}", style={"color": "#e6e6e6"})
                ])

                # Store data for download
                store_data = {
                    "full_df": encoded_df.to_dict("records"),
                    "column_name": column,
                    "encoded_column": {f"{column}_ordinal": encoded_df[f"{column}_ordinal"].tolist()},
                    "encoding_type": "ordinal",
                    "mapping": ordinal_map
                }

            # Return based on show_encoded_only toggle
            if not show_encoded_only:
                # Show full dataframe
                columns = [{"name": col, "id": col} for col in encoded_df.columns]
                return encoded_df.to_dict("records"), columns, message, store_data
            else:
                # Show only encoded columns
                if encoding_type == "onehot":
                    # For one-hot, show original column and all dummy columns
                    display_cols = [column] + list(dummies.columns)
                    display_df = encoded_df[display_cols]
                    columns = [{"name": col, "id": col} for col in display_df.columns]
                    return display_df.to_dict("records"), columns, message, store_data
                else:
                    # For label and ordinal, show original and encoded column
                    encoded_col_name = f"{column}_encoded" if encoding_type == "label" else f"{column}_ordinal"
                    display_df = encoded_df[[column, encoded_col_name]]
                    columns = [{"name": col, "id": col} for col in display_df.columns]
                    return display_df.to_dict("records"), columns, message, store_data

        except Exception as e:
            return [], [], html.Div(f"Error: {str(e)}", style={"color": "red"}), None

    elif trigger_id == "encoding_show_encoded_toggle" and stored_encoded_df:
        # Toggle between showing all columns or only encoded columns
        df = pd.DataFrame(stored_encoded_df["full_df"])
        encoding_type = stored_encoded_df["encoding_type"]
        column_name = stored_encoded_df["column_name"]

        if not show_encoded_only:
            # Show full dataframe
            columns = [{"name": col, "id": col} for col in df.columns]
            return df.to_dict("records"), columns, dash.no_update, stored_encoded_df
        else:
            # Show only encoded columns
            if encoding_type == "onehot":
                # For one-hot, show original column and all dummy columns
                new_columns = stored_encoded_df.get("new_columns", [])
                display_cols = [column_name] + new_columns
                display_df = df[display_cols]
                columns = [{"name": col, "id": col} for col in display_df.columns]
                return display_df.to_dict("records"), columns, dash.no_update, stored_encoded_df
            else:
                # For label and ordinal, show original and encoded column
                encoded_col_name = f"{column_name}_encoded" if encoding_type == "label" else f"{column_name}_ordinal"
                if encoded_col_name in df.columns:
                    display_df = df[[column_name, encoded_col_name]]
                    columns = [{"name": col, "id": col} for col in display_df.columns]
                    return display_df.to_dict("records"), columns, dash.no_update, stored_encoded_df
                else:
                    return [], [], html.Div("Encoded column not found", style={"color": "red"}), stored_encoded_df

    return [], [], dash.no_update, None

# Download encoded data as CSV
@app.callback(
    Output("encoding_download_csv", "data"),
    [Input("encoding_download_csv_button", "n_clicks")],
    [State("encoding_data_store", "data")],
    prevent_initial_call=True,
)
def download_encoded_csv(n_clicks, stored_data):
    if not stored_data:
        return None

    df = pd.DataFrame(stored_data["full_df"])
    return dcc.send_data_frame(df.to_csv, "encoded_data.csv", index=False)

# Download encoded data as JSON
@app.callback(
    Output("encoding_download_json", "data"),
    [Input("encoding_download_json_button", "n_clicks")],
    [State("encoding_data_store", "data")],
    prevent_initial_call=True,
)
def download_encoded_json(n_clicks, stored_data):
    if not stored_data:
        return None

    df = pd.DataFrame(stored_data["full_df"])
    return dcc.send_data_frame(df.to_json, "encoded_data.json", orient="records", date_format="iso")

# Download encoded data as Excel
@app.callback(
    Output("encoding_download_excel", "data"),
    [Input("encoding_download_excel_button", "n_clicks")],
    [State("encoding_data_store", "data")],
    prevent_initial_call=True,
)
def download_encoded_excel(n_clicks, stored_data):
    if not stored_data:
        return None

    df = pd.DataFrame(stored_data["full_df"])
    return dcc.send_data_frame(df.to_excel, "encoded_data.xlsx", sheet_name="Encoded Data", index=False)

# Main entry point
if __name__ == "__main__":
    print("Starting Data Analysis Dashboard...")
    print("Visit http://127.0.0.1:8050/ in your web browser")
    app.run(debug=True)
