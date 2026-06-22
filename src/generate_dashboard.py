"""
FASE 9b - Dashboard Generation
Generates an interactive HTML dashboard for data analysts.
"""

import os
import json
import pandas as pd
from src.logger import setup_logger

logger = setup_logger("DashboardGeneration")

class DashboardGenerator:
    def __init__(self, tables_dir="results/final_tables", output_dir="results"):
        self.tables_dir = tables_dir
        self.output_dir = output_dir
        
    def run(self):
        logger.info("Generating Interactive HTML Dashboard...")
        
        # Load Data
        perf_df = pd.DataFrame()
        rob_df = pd.DataFrame()
        cont_df = pd.DataFrame()
        
        perf_path = os.path.join(self.tables_dir, "final_performance.csv")
        if os.path.exists(perf_path):
            perf_df = pd.read_csv(perf_path)
            
        rob_path = os.path.join(self.tables_dir, "final_robustness.csv")
        if os.path.exists(rob_path):
            rob_df = pd.read_csv(rob_path)
            
        cont_path = os.path.join(self.tables_dir, "final_contribution.csv")
        if os.path.exists(cont_path):
            cont_df = pd.read_csv(cont_path)
            
        # Extract metrics for plotting
        models = perf_df['model'].tolist() if not perf_df.empty else []
        f1_scores = perf_df['f1_score_mean'].tolist() if 'f1_score_mean' in perf_df.columns else []
        accuracies = perf_df['accuracy_mean'].tolist() if 'accuracy_mean' in perf_df.columns else []
        
        # Robustness data
        r_models = rob_df['model'].tolist() if not rob_df.empty else []
        r_f1_mean = rob_df['f1_score_robust_mean'].tolist() if 'f1_score_robust_mean' in rob_df.columns else []
        r_f1_std = rob_df['f1_score_robust_std'].tolist() if 'f1_score_robust_std' in rob_df.columns else []
        r_f1_min = rob_df['f1_score_robust_min'].tolist() if 'f1_score_robust_min' in rob_df.columns else []
        r_f1_max = rob_df['f1_score_robust_max'].tolist() if 'f1_score_robust_max' in rob_df.columns else []
        
        # KPI calculations
        best_model = perf_df.loc[perf_df['f1_score_mean'].idxmax()]['model'] if not perf_df.empty and 'f1_score_mean' in perf_df.columns else "N/A"
        best_f1 = perf_df['f1_score_mean'].max() if not perf_df.empty and 'f1_score_mean' in perf_df.columns else 0.0
        dpbl_cont = cont_df[cont_df['metric'] == 'f1_score']['diff'].values[0] if not cont_df.empty and 'diff' in cont_df.columns and not cont_df[cont_df['metric'] == 'f1_score'].empty else 0.0
        robust_ci = rob_df['f1_score_robust_ci95'].values[0] if not rob_df.empty and 'f1_score_robust_ci95' in rob_df.columns else 0.0

        # Table HTML processing
        perf_html = perf_df.to_html(index=False, classes='data-table', border=0) if not perf_df.empty else '<p>No data available</p>'
        rob_html = rob_df.to_html(index=False, classes='data-table', border=0) if not rob_df.empty else '<p>No data available</p>'
        cont_html = cont_df.to_html(index=False, classes='data-table', border=0) if not cont_df.empty else '<p>No data available</p>'

        # HTML Template
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WESAD Stress Detection Dashboard</title>
    <!-- Plotly CDN -->
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;600&family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-main: #1a1b26; /* Tokyo Night Dark */
            --bg-card: #24283b;
            --text-main: #a9b1d6;
            --text-heading: #c0caf5;
            --accent-1: #7aa2f7;
            --accent-2: #bb9af7;
            --accent-3: #f7768e;
            --accent-4: #9ece6a;
            --border: #414868;
        }}
        body {{
            background-color: var(--bg-main);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 20px 40px;
        }}
        h1, h2, h3 {{
            color: var(--text-heading);
            font-family: 'Fira Code', monospace;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid var(--border);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .kpi-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .kpi-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-left: 4px solid var(--accent-1);
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }}
        .kpi-value {{
            font-size: 2em;
            font-weight: bold;
            color: var(--accent-2);
            margin: 10px 0;
            font-family: 'Fira Code', monospace;
        }}
        .kpi-label {{
            color: var(--text-main);
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .tabs {{
            display: flex;
            border-bottom: 2px solid var(--border);
            margin-bottom: 20px;
        }}
        .tab {{
            padding: 10px 20px;
            cursor: pointer;
            color: var(--text-main);
            border-bottom: 2px solid transparent;
            margin-bottom: -2px;
            transition: all 0.3s;
        }}
        .tab:hover {{
            color: var(--accent-1);
        }}
        .tab.active {{
            color: var(--accent-1);
            border-bottom-color: var(--accent-1);
            font-weight: bold;
        }}
        .tab-content {{
            display: none;
            animation: fadeIn 0.5s;
        }}
        .tab-content.active {{
            display: block;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        .full-width {{
            grid-column: 1 / -1;
        }}
        table.data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
            text-align: left;
        }}
        table.data-table th, table.data-table td {{
            padding: 12px;
            border-bottom: 1px solid var(--border);
        }}
        table.data-table th {{
            color: var(--accent-1);
            font-family: 'Fira Code', monospace;
            background-color: rgba(65, 72, 104, 0.3);
        }}
        table.data-table tr:hover {{
            background-color: rgba(65, 72, 104, 0.5);
        }}
    </style>
</head>
<body>

    <div class="header">
        <div>
            <h1>WESAD Stress Detection Analysis</h1>
            <p>Interactive Performance & Robustness Dashboard</p>
        </div>
        <div>
            <h3>Theme: Tokyo Night</h3>
        </div>
    </div>

    <!-- KPI Cards -->
    <div class="kpi-container">
        <div class="kpi-card" style="border-left-color: var(--accent-1);">
            <div class="kpi-label">Best Model</div>
            <div class="kpi-value">{best_model}</div>
        </div>
        <div class="kpi-card" style="border-left-color: var(--accent-2);">
            <div class="kpi-label">Best F1-Score</div>
            <div class="kpi-value">{best_f1:.4f}</div>
        </div>
        <div class="kpi-card" style="border-left-color: var(--accent-3);">
            <div class="kpi-label">DPBL Contribution (F1)</div>
            <div class="kpi-value">+{dpbl_cont:.4f}</div>
        </div>
        <div class="kpi-card" style="border-left-color: var(--accent-4);">
            <div class="kpi-label">HSSL+DPBL Robustness (CI)</div>
            <div class="kpi-value">±{robust_ci:.4f}</div>
        </div>
    </div>

    <!-- Tabs Navigation -->
    <div class="tabs">
        <div class="tab active" onclick="openTab(event, 'tab-overview')">Overview</div>
        <div class="tab" onclick="openTab(event, 'tab-performance')">Performance</div>
        <div class="tab" onclick="openTab(event, 'tab-robustness')">Robustness</div>
        <div class="tab" onclick="openTab(event, 'tab-statistics')">Statistical Validation</div>
    </div>

    <!-- Tab Contents -->
    <div id="tab-overview" class="tab-content active">
        <div class="grid">
            <div class="card full-width">
                <h2>Confusion Matrices (Per Model)</h2>
                <div style="display:flex; justify-content:space-around;">
                    <img src="figures/cm_rf.png" style="width:24%; max-width:300px; border-radius:8px;" onerror="this.style.display='none'">
                    <img src="figures/cm_cnn.png" style="width:24%; max-width:300px; border-radius:8px;" onerror="this.style.display='none'">
                    <img src="figures/cm_hssl.png" style="width:24%; max-width:300px; border-radius:8px;" onerror="this.style.display='none'">
                    <img src="figures/cm_hssl+dpbl.png" style="width:24%; max-width:300px; border-radius:8px;" onerror="this.style.display='none'">
                </div>
            </div>
        </div>
        <div class="grid">
            <div class="card full-width">
                <h2>Model Performance Comparison (Metrics)</h2>
                <div id="perfRadarChart" style="height: 500px;"></div>
            </div>
        </div>
    </div>

    <div id="tab-performance" class="tab-content">
        <div class="grid">
            <div class="card full-width">
                <h2>Per-Subject F1-Score Heatmap</h2>
                <div style="display:flex; justify-content:center; margin-bottom: 20px;">
                    <img src="figures/per_subject_f1.png" style="width:80%; max-width:800px; border-radius:8px;" onerror="this.style.display='none'">
                </div>
            </div>
            <div class="card full-width">
                <h2>F1-Score and Accuracy Bar Chart</h2>
                <div id="perfChart"></div>
            </div>
            <div class="card full-width">
                <h2>Performance Metrics Table</h2>
                <div style="overflow-x:auto;">
                    {perf_html}
                </div>
            </div>
        </div>
    </div>

    <div id="tab-robustness" class="tab-content">
        <div class="grid">
            <div class="card full-width">
                <h2>Robustness (F1-Score Stability with 95% CI)</h2>
                <div id="robustChart"></div>
            </div>
            <div class="card full-width">
                <h2>Robustness Details Table</h2>
                <div style="overflow-x:auto;">
                    {rob_html}
                </div>
            </div>
        </div>
    </div>

    <div id="tab-statistics" class="tab-content">
        <div class="grid">
            <div class="card full-width">
                <h2>Statistical Contribution Analysis (HSSL vs HSSL+DPBL)</h2>
                <div style="overflow-x:auto;">
                    {cont_html}
                </div>
            </div>
        </div>
    </div>

    <script>
        const layoutConfig = {{
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: {{ color: '#a9b1d6', family: 'Inter' }},
            xaxis: {{ gridcolor: '#414868', zerolinecolor: '#414868' }},
            yaxis: {{ gridcolor: '#414868', zerolinecolor: '#414868' }},
            margin: {{ t: 40, b: 40, l: 50, r: 20 }}
        }};

        // Tab switching logic
        function openTab(evt, tabName) {{
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {{
                tabcontent[i].className = tabcontent[i].className.replace(" active", "");
            }}
            tablinks = document.getElementsByClassName("tab");
            for (i = 0; i < tablinks.length; i++) {{
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }}
            document.getElementById(tabName).className += " active";
            evt.currentTarget.className += " active";
            
            // Trigger Plotly resize when tab becomes visible
            window.dispatchEvent(new Event('resize'));
        }}

        // Data for Performance Chart
        const models = {json.dumps(models)};
        const f1_scores = {json.dumps(f1_scores)};
        const accuracies = {json.dumps(accuracies)};
        
        // Extract all metrics for Radar Chart
        const metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'];
        const radarData = [];
        const colors = ['#7aa2f7', '#bb9af7', '#f7768e', '#9ece6a'];
        
        // Extract data from Python variables that will be injected
        const perfDataRaw = {perf_df.to_json(orient='records') if not perf_df.empty else '[]'};
        
        if (perfDataRaw.length > 0) {{
            perfDataRaw.forEach((row, i) => {{
                radarData.push({{
                    type: 'scatterpolar',
                    r: metrics.map(m => row[m+'_mean']),
                    theta: ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'],
                    fill: 'toself',
                    name: row.model,
                    line: {{ color: colors[i % colors.length] }}
                }});
            }});
            
            const radarLayout = Object.assign({{}}, layoutConfig, {{
                polar: {{
                    radialaxis: {{
                        visible: true,
                        range: [0, 1],
                        gridcolor: '#414868',
                        linecolor: '#414868'
                    }},
                    angularaxis: {{
                        gridcolor: '#414868',
                        linecolor: '#414868'
                    }},
                    bgcolor: 'transparent'
                }},
                showlegend: true
            }});
            
            Plotly.newPlot('perfRadarChart', radarData, radarLayout);
        }}

        const trace1 = {{
            x: models,
            y: f1_scores,
            name: 'F1-Score (Mean)',
            type: 'bar',
            marker: {{ color: '#7aa2f7' }}
        }};
        const trace2 = {{
            x: models,
            y: accuracies,
            name: 'Accuracy (Mean)',
            type: 'bar',
            marker: {{ color: '#bb9af7' }}
        }};

        const perfLayout = Object.assign({{}}, layoutConfig, {{ barmode: 'group' }});
        Plotly.newPlot('perfChart', [trace1, trace2], perfLayout);

        // Data for Robustness Chart
        const robDataRaw = {rob_df.to_json(orient='records') if not rob_df.empty else '[]'};
        if (robDataRaw.length > 0) {{
            const r_models = robDataRaw.map(r => r.model);
            const r_f1_mean = robDataRaw.map(r => r.f1_score_robust_mean);
            const r_ci = robDataRaw.map(r => r.f1_score_robust_ci95);
            
            const robustTrace = {{
                x: r_models,
                y: r_f1_mean,
                name: 'F1-Score (Robust)',
                type: 'scatter',
                mode: 'markers+lines',
                marker: {{ color: '#f7768e', size: 12 }},
                line: {{ color: '#f7768e', width: 3 }},
                error_y: {{
                    type: 'data',
                    array: r_ci,
                    visible: true,
                    color: '#9ece6a',
                    thickness: 2,
                    width: 8
                }}
            }};

            const robustLayout = Object.assign({{}}, layoutConfig, {{
                yaxis: {{ title: 'F1-Score', gridcolor: '#414868', range: [0, 1.05] }},
                title: 'F1-Score Stability (95% CI)'
            }});
            
            Plotly.newPlot('robustChart', [robustTrace], robustLayout);
        }}
    </script>
</body>
</html>
"""
        
        out_path = os.path.join(self.output_dir, "interactive_dashboard.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        logger.info(f"Dashboard generated -> {out_path}")

if __name__ == "__main__":
    generator = DashboardGenerator()
    generator.run()