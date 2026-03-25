/**
 * AlignScope — Metrics Panel
 *
 * Updates the right sidebar with real-time alignment metrics:
 * - Overall alignment score with progress bar
 * - Per-team: role stability, coalitions, defectors, score
 * - Alignment history sparkline
 *
 * Dynamically supports N teams from config — no hardcoded team names.
 */

const MetricsPanel = {
    config: null,
    sparklineHistory: [],
    maxSparklinePoints: 200,

    init(config) {
        this.config = config;
        this.sparklineHistory = [];
    },

    update(data) {
        if (!data.metrics) return;

        const metrics = data.metrics;
        const scores = data.team_scores || {};

        // Overall alignment
        const overall = metrics.overall_alignment_score;
        if (overall !== undefined && overall !== null) {
            document.getElementById('metric-overall-value').textContent = overall.toFixed(3);
            document.getElementById('metric-overall-bar').style.width = (overall * 100) + '%';

            const barEl = document.getElementById('metric-overall-bar');
            if (overall > 0.8) barEl.style.background = 'var(--color-success)';
            else if (overall > 0.5) barEl.style.background = 'var(--color-accent)';
            else barEl.style.background = 'var(--color-defection)';
        }

        // Team metrics — iterate over all teams from config
        const teamMetrics = metrics.team_metrics || {};
        const numTeams = (this.config && this.config.teams) ? this.config.teams.length : 2;

        for (let i = 0; i < numTeams; i++) {
            const teamData = teamMetrics[String(i)] || teamMetrics[i];
            const teamScore = scores[String(i)] ?? scores[i] ?? 0;
            this.updateTeam(`team-${i}`, teamData, teamScore);
        }

        // Sparkline
        this.sparklineHistory.push(overall);
        if (this.sparklineHistory.length > this.maxSparklinePoints) {
            this.sparklineHistory.shift();
        }
        this.drawSparkline();

        // Update agent detail if one is selected
        if (AlignScope.state.selectedAgent !== null) {
            AlignScope.selectAgent(AlignScope.state.selectedAgent);
        }
    },

    updateTeam(prefix, teamData, score) {
        if (!teamData) return;

        // Legacy mapping for cooperative built-ins
        const mappedData = {
            ...teamData,
            role_stability: teamData.avg_role_stability,
            coalitions: teamData.active_coalitions,
            defectors: teamData.defector_count,
            score: score
        };

        const metricsConfig = this.config?.metrics || [];
        metricsConfig.forEach(m => {
            const el = document.getElementById(`${prefix}-metric-${m.id}`);
            if (!el) return;

            let val = mappedData[m.id];
            if (val === undefined) val = 0; // fallback if missing

            // special formatting
            if (typeof val === 'number') {
                if (Number.isInteger(val)) {
                    el.textContent = val.toString();
                } else {
                    el.textContent = val.toFixed(m.id === 'role_stability' ? 4 : 2);
                }
            } else {
                el.textContent = String(val);
            }

            // Colors for defectors
            if (m.id === 'defectors') {
                if (val > 0) el.style.color = 'var(--color-defection)';
                else el.style.color = 'var(--color-success)';
            }
            
            // role_stability mini-bar
            if (m.id === 'role_stability') {
                const stabBarEl = document.getElementById(`${prefix}-stability-bar`);
                if (stabBarEl) {
                    stabBarEl.style.width = (Math.max(0, Math.min(1, val)) * 100) + '%';
                }
            }
        });
    },

    drawSparkline() {
        const canvas = document.getElementById('alignment-sparkline');
        if (!canvas) return;

        const container = canvas.parentElement;

        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        const rect = container.getBoundingClientRect();
        const data = this.sparklineHistory;

        // Dynamic width: min size is container width, otherwise 2px per data point
        const desiredWidth = Math.max(rect.width, Math.max(0, data.length) * 2);
        const fixedHeight = 80; // Keep height locked to 80px to prevent infinite container growth loop

        canvas.width = desiredWidth * dpr;
        canvas.height = fixedHeight * dpr;
        canvas.style.width = desiredWidth + 'px';
        canvas.style.height = fixedHeight + 'px';
        ctx.scale(dpr, dpr);

        const w = desiredWidth;
        const h = fixedHeight;
        const padding = { top: 4, bottom: 4, left: 2, right: 2 };
        const plotW = w - padding.left - padding.right;
        const plotH = h - padding.top - padding.bottom;

        ctx.clearRect(0, 0, w, h);

        if (data.length < 2) return;

        const min = Math.min(...data) * 0.95;
        const max = Math.max(...data) * 1.05;
        const range = max - min || 1;

        const xScale = (i) => padding.left + (i / (data.length - 1)) * plotW;
        const yScale = (v) => padding.top + plotH - ((v - min) / range) * plotH;

        // Fill area under the line
        ctx.beginPath();
        ctx.moveTo(xScale(0), yScale(data[0]));
        for (let i = 1; i < data.length; i++) {
            ctx.lineTo(xScale(i), yScale(data[i]));
        }
        ctx.lineTo(xScale(data.length - 1), padding.top + plotH);
        ctx.lineTo(xScale(0), padding.top + plotH);
        ctx.closePath();

        const gradient = ctx.createLinearGradient(0, padding.top, 0, padding.top + plotH);
        gradient.addColorStop(0, 'rgba(109, 158, 235, 0.15)');
        gradient.addColorStop(1, 'rgba(109, 158, 235, 0.02)');
        ctx.fillStyle = gradient;
        ctx.fill();

        // Draw line
        ctx.beginPath();
        ctx.moveTo(xScale(0), yScale(data[0]));
        for (let i = 1; i < data.length; i++) {
            ctx.lineTo(xScale(i), yScale(data[i]));
        }
        ctx.strokeStyle = '#6d9eeb';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Draw current value dot
        const lastX = xScale(data.length - 1);
        const lastY = yScale(data[data.length - 1]);
        ctx.beginPath();
        ctx.arc(lastX, lastY, 3, 0, Math.PI * 2);
        ctx.fillStyle = '#6d9eeb';
        ctx.fill();

        // Min/max labels
        ctx.font = '9px "JetBrains Mono"';
        ctx.fillStyle = '#4b5563';
        ctx.textAlign = 'right';
        ctx.fillText(max.toFixed(2), w - 2, padding.top + 8);
        ctx.fillText(min.toFixed(2), w - 2, padding.top + plotH);
    },

    reset() {
        this.sparklineHistory = [];

        document.getElementById('metric-overall-value').textContent = '—';
        document.getElementById('metric-overall-bar').style.width = '0%';

        const numTeams = (this.config && this.config.teams) ? this.config.teams.length : 2;
        const metricsConfig = this.config?.metrics || [];
        
        for (let i = 0; i < numTeams; i++) {
            const prefix = `team-${i}`;
            metricsConfig.forEach(m => {
                const el = document.getElementById(`${prefix}-metric-${m.id}`);
                if (el) {
                    el.textContent = m.id === 'defectors' || m.id === 'coalitions' ? '0' : '—';
                    if (m.id === 'defectors') el.style.color = '';
                }
                if (m.id === 'role_stability') {
                    const stabBarEl = document.getElementById(`${prefix}-stability-bar`);
                    if (stabBarEl) stabBarEl.style.width = '0%';
                }
            });
        }

        const canvas = document.getElementById('alignment-sparkline');
        if (canvas) {
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
    },
};
