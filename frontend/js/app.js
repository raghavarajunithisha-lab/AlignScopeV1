/**
 * AlignScope — Main Application Controller
 *
 * Manages WebSocket connection, state, and coordinates
 * the three dashboard panels (topology, timeline, metrics).
 *
 * Dynamically builds UI elements based on the config message
 * received from the backend — supports any number of teams and roles.
 */

const AlignScope = {
    ws: null,
    config: null,
    state: {
        tick: 0,
        agents: [],
        relationships: [],
        events: [],
        metrics: null,
        teamScores: {},
        alignmentHistory: [],
        isConnected: false,
        selectedAgent: null,
    },

    // Default color palette for teams (expandable)
    teamColors: [
        '#6d9eeb', '#e8925a', '#4abe7d', '#b06ec7',
        '#d4a843', '#dc4a4a', '#5bc0de', '#8cc152',
    ],

    // Role shape symbols (unicode) for the legend
    roleShapeSymbols: ['●', '◆', '■', '▲', '★', '⬟', '⬠', '◈'],

    init() {
        this.connectWebSocket();
        this.bindUI();
        window.addEventListener('resize', () => {
            TopologyGraph.resize();
            TimelineView.resize();
        });
    },

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        this.ws = new WebSocket(wsUrl);
        this.updateConnectionStatus('connecting');

        this.ws.onopen = () => {
            this.updateConnectionStatus('connected');
        };

        this.ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            this.handleMessage(msg);
        };

        this.ws.onclose = () => {
            this.updateConnectionStatus('disconnected');
            setTimeout(() => this.connectWebSocket(), 2000);
        };

        this.ws.onerror = () => {
            this.updateConnectionStatus('disconnected');
        };
    },

    handleMessage(msg) {
        switch (msg.type) {
            case 'config':
                this.config = msg.data;
                this.buildDynamicUI(msg.data);
                try { TopologyGraph.init(this.config); } catch(e) { console.error('TopologyGraph.init error:', e); }
                try { TimelineView.init(this.config); } catch(e) { console.error('TimelineView.init error:', e); }
                try { MetricsPanel.init(this.config); } catch(e) { console.error('MetricsPanel.init error:', e); }
                break;

            case 'tick':
                this.processTick(msg.data);
                break;

            case 'restart':
                this.resetState();
                break;

            case 'episode_complete':
                this.handleEpisodeComplete(msg.data);
                break;
        }
    },

    /**
     * Dynamically build UI elements from the config payload.
     * This makes the dashboard work with any team/role configuration.
     */
    buildDynamicUI(config) {
        // Update panel titles based on paradigm if available
        if (config.paradigm && config.paradigm.environment) {
            const envName = config.paradigm.environment.charAt(0).toUpperCase() + config.paradigm.environment.slice(1);
            const metricsTitle = document.getElementById('metrics-panel-title');
            if (metricsTitle) metricsTitle.textContent = `${envName} Metrics`;
        }

        // Build team legend
        const teamLegend = document.getElementById('legend-teams');
        // Keep the title, remove old items
        const teamTitle = teamLegend.querySelector('.legend-title');
        teamLegend.innerHTML = '';
        teamLegend.appendChild(teamTitle);

        (config.teams || []).forEach((team, idx) => {
            const color = team.color || this.teamColors[idx % this.teamColors.length];
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML = `
                <span class="legend-dot" style="background: ${color}"></span>
                <span>${team.name}</span>
            `;
            teamLegend.appendChild(item);
        });

        // Build role legend
        const roleLegend = document.getElementById('legend-roles');
        const roleTitle = roleLegend.querySelector('.legend-title');
        roleLegend.innerHTML = '';
        roleLegend.appendChild(roleTitle);

        (config.roles || []).forEach((role, idx) => {
            const symbol = this.roleShapeSymbols[idx % this.roleShapeSymbols.length];
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML = `
                <span class="legend-shape">${symbol}</span>
                <span>${role.charAt(0).toUpperCase() + role.slice(1)}</span>
            `;
            roleLegend.appendChild(item);
        });

        // Build team metric sections in the sidebar
        const container = document.getElementById('team-metrics-container');
        container.innerHTML = '';

        (config.teams || []).forEach((team, idx) => {
            const color = team.color || this.teamColors[idx % this.teamColors.length];
            const teamId = `team-${idx}`;

            const section = document.createElement('div');
            section.className = 'metric-group';
            section.id = `${teamId}-metrics`;
            
            let html = `
                <div class="metric-group-header">
                    <span class="team-dot" style="background: ${color}"></span>
                    <span class="metric-group-title">${team.name} (Team ${idx})</span>
                </div>
            `;
            
            // Generate standard metrics from config
            (config.metrics || []).forEach(m => {
                html += `
                <div class="metric-row">
                    <span class="metric-label">${m.label}</span>
                    <span class="metric-val mono" id="${teamId}-metric-${m.id}">—</span>
                </div>`;
                
                // Preserve mini-bar explicitly for role_stability if present
                if (m.id === 'role_stability') {
                    html += `
                    <div class="metric-mini-bar-container">
                        <div class="metric-mini-bar" id="${teamId}-stability-bar" style="background: ${color}"></div>
                    </div>`;
                }
            });

            section.innerHTML = html;
            container.appendChild(section);
        });
    },

    processTick(data) {
        this.state.tick = data.tick || this.state.tick;
        
        if (data.agents && data.agents.length > 0) {
            this.state.agents = data.agents;
        }
        if (data.relationships && data.relationships.length > 0) {
            this.state.relationships = data.relationships;
        }
        
        if (data.team_scores && Object.keys(data.team_scores).length > 0) {
            this.state.teamScores = { ...this.state.teamScores, ...data.team_scores };
        }
        
        // Merge metrics seamlessly to support partial update logging
        if (data.metrics) {
            if (!this.state.metrics) this.state.metrics = { agent_metrics: {}, team_metrics: {}, pair_metrics: [] };
            
            if (data.metrics.agent_metrics) {
                Object.keys(data.metrics.agent_metrics).forEach(aId => {
                    this.state.metrics.agent_metrics[aId] = {
                        ...(this.state.metrics.agent_metrics[aId] || {}),
                        ...data.metrics.agent_metrics[aId]
                    };
                });
            }
            if (data.metrics.team_metrics) {
                Object.keys(data.metrics.team_metrics).forEach(tId => {
                    this.state.metrics.team_metrics[tId] = {
                        ...(this.state.metrics.team_metrics[tId] || {}),
                        ...data.metrics.team_metrics[tId]
                    };
                });
            }
            if (data.metrics.overall_alignment_score !== undefined) {
                this.state.metrics.overall_alignment_score = data.metrics.overall_alignment_score;
                this.state.alignmentHistory.push({
                    tick: data.tick,
                    score: data.metrics.overall_alignment_score,
                });
            }
        }

        if (data.events && data.events.length > 0) {
            this.state.events.push(...data.events);
        }

        // Alignment history logic was moved to merge section above

        this.updateHeader(data);

        try { TopologyGraph.update(data); } catch(e) { console.error('TopologyGraph error:', e); }
        try { TimelineView.update(data); } catch(e) { console.error('TimelineView error:', e); }
        try { MetricsPanel.update(data); } catch(e) { console.error('MetricsPanel error:', e); }
    },

    updateHeader(data) {
        document.getElementById('tick-value').textContent = data.tick;

        if (data.metrics) {
            const score = data.metrics.overall_alignment_score;
            const el = document.getElementById('alignment-value');
            el.textContent = score.toFixed(3);
            if (score > 0.8) el.style.color = 'var(--color-success)';
            else if (score > 0.5) el.style.color = 'var(--text-primary)';
            else el.style.color = 'var(--color-defection)';
        }

        document.getElementById('events-count').textContent = this.state.events.length;
    },

    updateConnectionStatus(status) {
        const el = document.getElementById('connection-status');
        const textEl = el.querySelector('.connection-text');

        el.className = 'connection-status ' + status;

        switch (status) {
            case 'connected':
                textEl.textContent = 'Live';
                this.state.isConnected = true;
                break;
            case 'disconnected':
                textEl.textContent = 'Disconnected';
                this.state.isConnected = false;
                break;
            default:
                textEl.textContent = 'Connecting…';
        }
    },

    bindUI() {
        document.getElementById('btn-restart').addEventListener('click', () => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ action: 'restart' }));
            }
        });

        document.getElementById('btn-close-detail').addEventListener('click', () => {
            document.getElementById('agent-detail').classList.add('hidden');
            this.state.selectedAgent = null;
            TopologyGraph.clearSelection();
        });
    },

    resetState() {
        this.state.tick = 0;
        this.state.agents = [];
        this.state.relationships = [];
        this.state.events = [];
        this.state.metrics = null;
        this.state.teamScores = {};
        this.state.alignmentHistory = [];
        this.state.selectedAgent = null;

        document.getElementById('tick-value').textContent = '0';
        document.getElementById('alignment-value').textContent = '—';
        document.getElementById('events-count').textContent = '0';
        document.getElementById('agent-detail').classList.add('hidden');

        TopologyGraph.reset();
        TimelineView.reset();
        MetricsPanel.reset();
    },

    handleEpisodeComplete(data) {
        console.log('Episode complete:', data);
    },

    /**
     * Get team name from config by team index.
     */
    getTeamName(teamIdx) {
        if (this.config && this.config.teams && this.config.teams[teamIdx]) {
            return this.config.teams[teamIdx].name;
        }
        return `Team ${teamIdx}`;
    },

    /**
     * Get team color from config by team index.
     */
    getTeamColor(teamIdx) {
        if (this.config && this.config.teams && this.config.teams[teamIdx]) {
            return this.config.teams[teamIdx].color || this.teamColors[teamIdx % this.teamColors.length];
        }
        return this.teamColors[teamIdx % this.teamColors.length];
    },

    selectAgent(agentId) {
        this.state.selectedAgent = agentId;
        const agent = this.state.agents.find(a => a.agent_id === agentId);
        if (!agent) return;

        const metrics = this.state.metrics?.agent_metrics?.[agentId];

        document.getElementById('detail-agent-id').textContent = agentId;
        document.getElementById('detail-team').textContent = this.getTeamName(agent.team);
        document.getElementById('detail-role').textContent = agent.role;
        document.getElementById('detail-stability').textContent =
            metrics ? metrics.role_stability.toFixed(4) : '—';
        document.getElementById('detail-coalition').textContent =
            agent.coalition_id >= 0 ? `Coalition ${agent.coalition_id}` : 'None (loner)';

        const statusEl = document.getElementById('detail-status');
        if (agent.is_defector) {
            statusEl.textContent = 'Defected';
            statusEl.style.color = 'var(--color-defection)';
        } else {
            statusEl.textContent = 'Active';
            statusEl.style.color = 'var(--color-success)';
        }

        document.getElementById('agent-detail').classList.remove('hidden');
    },
};

// Boot
document.addEventListener('DOMContentLoaded', () => AlignScope.init());
