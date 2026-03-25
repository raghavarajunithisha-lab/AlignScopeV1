/**
 * AlignScope — Topology Graph (D3.js Force-Directed)
 *
 * Renders agents as nodes and their relationships as edges.
 * Coalition clusters form visually via D3 force simulation parameters.
 *
 * Nodes: colored by team (from config), sized by role stability
 * Edges: opacity = reciprocity strength, width = total interactions
 * Defectors: pulsing animation, pushed to periphery
 *
 * All team colors and role shapes are dynamically assigned from config.
 */

const TopologyGraph = {
    svg: null,
    simulation: null,
    config: null,
    width: 0,
    height: 0,
    nodes: [],
    links: [],
    nodeElements: null,
    linkElements: null,
    labelElements: null,

    // D3 symbol types palette — assigned dynamically to roles
    shapePalette: [
        d3.symbolCircle,
        d3.symbolDiamond,
        d3.symbolSquare,
        d3.symbolTriangle,
        d3.symbolStar,
        d3.symbolCross,
        d3.symbolWye,
    ],

    // Computed mappings (built from config)
    roleShapes: {},
    roleSize: {},

    init(config) {
        this.config = config;

        // Build role → shape mapping dynamically from config
        const roles = config.roles || [];
        this.roleShapes = {};
        this.roleSize = {};
        roles.forEach((role, idx) => {
            this.roleShapes[role] = this.shapePalette[idx % this.shapePalette.length];
            this.roleSize[role] = 55 + (idx % 4) * 5;  // slight size variation
        });

        const container = document.getElementById('topology-container');
        this.width = container.clientWidth;
        this.height = container.clientHeight;

        this.svg = d3.select('#topology-svg')
            .attr('width', this.width)
            .attr('height', this.height);

        // Clear any previous content
        this.svg.selectAll('*').remove();

        // Create layer groups (links behind nodes)
        this.svg.append('g').attr('class', 'links-layer');
        this.svg.append('g').attr('class', 'nodes-layer');
        this.svg.append('g').attr('class', 'labels-layer');

        // Initialize force simulation
        this.simulation = d3.forceSimulation()
            .force('charge', d3.forceManyBody()
                .strength(-120)
                .distanceMax(250))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('collision', d3.forceCollide().radius(22))
            .force('x', d3.forceX(this.width / 2).strength(0.03))
            .force('y', d3.forceY(this.height / 2).strength(0.03))
            .alphaDecay(0.01)
            .on('tick', () => this.ticked());

        this.updateBadge(0);
    },

    /**
     * Get team color from config or fallback to AlignScope palette.
     */
    getTeamColor(teamIdx) {
        return AlignScope.getTeamColor(teamIdx);
    },

    /**
     * Get a slightly darker team color for stroke.
     */
    getTeamStroke(teamIdx) {
        const color = this.getTeamColor(teamIdx);
        // Darken by mixing toward black
        return this._darkenColor(color, 0.2);
    },

    _darkenColor(hex, amount) {
        // Simple hex darkening
        let c = hex.replace('#', '');
        if (c.length === 3) c = c[0]+c[0]+c[1]+c[1]+c[2]+c[2];
        const r = Math.max(0, Math.round(parseInt(c.substr(0,2), 16) * (1 - amount)));
        const g = Math.max(0, Math.round(parseInt(c.substr(2,2), 16) * (1 - amount)));
        const b = Math.max(0, Math.round(parseInt(c.substr(4,2), 16) * (1 - amount)));
        return `rgb(${r},${g},${b})`;
    },

    update(data) {
        if (!this.svg) return;

        const agents = data.agents || [];
        const relationships = data.relationships || [];
        const metrics = data.metrics?.agent_metrics || {};
        const groupBy = this.config?.topology?.groupBy || 'team';

        // Build node data
        const nodeIds = new Set(agents.map(a => a.agent_id));
        this.nodes = agents.map(a => {
            const existing = this.simulation?.nodes()?.find(n => n.id === a.agent_id);
            const m = metrics[a.agent_id] || metrics[String(a.agent_id)] || {};
            return {
                id: a.agent_id,
                team: a[groupBy] !== undefined ? a[groupBy] : a.team,
                role: a.role,
                isDefector: a.is_defector,
                coalitionId: a.coalition_id,
                roleStability: m.role_stability || 1.0,
                x: existing?.x || this.width / 2 + (Math.random() - 0.5) * 100,
                y: existing?.y || this.height / 2 + (Math.random() - 0.5) * 100,
                vx: existing?.vx || 0,
                vy: existing?.vy || 0,
            };
        });

        // Build link data
        this.links = relationships
            .filter(r => r.weight > 0 && nodeIds.has(r.source) && nodeIds.has(r.target))
            .map(r => ({
                source: r.source,
                target: r.target,
                weight: r.weight,
                reciprocity: r.reciprocity,
                sameTeam: r.same_team,
            }));

        this.render();
        this.updateBadge(agents.length);

        this.simulation.nodes(this.nodes);
        this.simulation
            .force('link', d3.forceLink(this.links)
                .id(d => d.id)
                .distance(d => {
                    const base = 100;
                    return Math.max(40, base - d.weight * 5);
                })
                .strength(d => {
                    return d.sameTeam ? 0.3 + d.reciprocity * 0.4 : 0.05;
                }))
            .alpha(0.15)
            .restart();
    },

    render() {
        const linksLayer = this.svg.select('.links-layer');
        const nodesLayer = this.svg.select('.nodes-layer');
        const labelsLayer = this.svg.select('.labels-layer');

        // --- Links ---
        this.linkElements = linksLayer.selectAll('.link-line')
            .data(this.links, d => `${d.source.id || d.source}-${d.target.id || d.target}`);

        this.linkElements.exit().remove();

        const linkEnter = this.linkElements.enter()
            .append('line')
            .attr('class', 'link-line');

        this.linkElements = linkEnter.merge(this.linkElements)
            .attr('stroke-width', d => Math.max(0.5, Math.min(3, d.weight * 0.3)))
            .attr('stroke-opacity', d => Math.max(0.08, Math.min(0.6, d.reciprocity * 0.8)))
            .attr('stroke', d => d.sameTeam ? 'var(--text-muted)' : 'var(--border-subtle)');

        // --- Nodes ---
        this.nodeElements = nodesLayer.selectAll('.node-group')
            .data(this.nodes, d => d.id);

        this.nodeElements.exit().remove();

        const nodeEnter = this.nodeElements.enter()
            .append('g')
            .attr('class', 'node-group')
            .style('cursor', 'pointer')
            .on('click', (event, d) => {
                AlignScope.selectAgent(d.id);
                this.highlightAgent(d.id);
            })
            .call(d3.drag()
                .on('start', (event, d) => this.dragStarted(event, d))
                .on('drag', (event, d) => this.dragged(event, d))
                .on('end', (event, d) => this.dragEnded(event, d)));

        nodeEnter.append('path').attr('class', 'node-shape');

        this.nodeElements = nodeEnter.merge(this.nodeElements);

        // Update node appearance — colors and shapes from config
        this.nodeElements.select('.node-shape')
            .attr('d', d => {
                const size = this.roleSize[d.role] || 60;
                const scaledSize = size * (0.8 + d.roleStability * 0.6);
                const shape = this.roleShapes[d.role] || d3.symbolCircle;
                return d3.symbol().type(shape).size(scaledSize)();
            })
            .attr('fill', d => {
                if (d.isDefector) return 'var(--color-defection)';
                return this.getTeamColor(d.team);
            })
            .attr('stroke', d => {
                if (d.isDefector) return 'var(--color-defection)';
                return this.getTeamStroke(d.team);
            })
            .attr('stroke-width', 1.5)
            .attr('opacity', d => d.isDefector ? 0.7 : 0.9)
            .classed('node-defector', d => d.isDefector);

        // --- Labels ---
        this.labelElements = labelsLayer.selectAll('.node-label')
            .data(this.nodes, d => d.id);

        this.labelElements.exit().remove();

        const labelEnter = this.labelElements.enter()
            .append('text')
            .attr('class', 'node-label')
            .attr('text-anchor', 'middle')
            .attr('dy', -12);

        this.labelElements = labelEnter.merge(this.labelElements)
            .text(d => d.id);
    },

    ticked() {
        if (this.linkElements) {
            this.linkElements
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
        }

        if (this.nodeElements) {
            this.nodeElements
                .attr('transform', d => {
                    d.x = Math.max(20, Math.min(this.width - 20, d.x));
                    d.y = Math.max(20, Math.min(this.height - 20, d.y));
                    return `translate(${d.x},${d.y})`;
                });
        }

        if (this.labelElements) {
            this.labelElements
                .attr('x', d => d.x)
                .attr('y', d => d.y);
        }
    },

    dragStarted(event, d) {
        if (!event.active) this.simulation.alphaTarget(0.1).restart();
        d.fx = d.x;
        d.fy = d.y;
    },

    dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    },

    dragEnded(event, d) {
        if (!event.active) this.simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    },

    highlightAgent(agentId) {
        if (!this.nodeElements) return;

        this.nodeElements.select('.node-shape')
            .attr('stroke-width', d => d.id === agentId ? 3 : 1.5)
            .attr('stroke', d => {
                if (d.id === agentId) return 'var(--text-primary)';
                if (d.isDefector) return 'var(--color-defection)';
                return this.getTeamStroke(d.team);
            });

        if (this.linkElements) {
            this.linkElements
                .attr('stroke-opacity', d => {
                    const srcId = d.source.id ?? d.source;
                    const tgtId = d.target.id ?? d.target;
                    if (srcId === agentId || tgtId === agentId) {
                        return 0.8;
                    }
                    return Math.max(0.05, Math.min(0.3, d.reciprocity * 0.4));
                })
                .attr('stroke-width', d => {
                    const srcId = d.source.id ?? d.source;
                    const tgtId = d.target.id ?? d.target;
                    if (srcId === agentId || tgtId === agentId) {
                        return Math.max(1.5, d.weight * 0.5);
                    }
                    return Math.max(0.5, Math.min(3, d.weight * 0.3));
                });
        }
    },

    clearSelection() {
        if (!this.nodeElements) return;
        this.nodeElements.select('.node-shape')
            .attr('stroke-width', 1.5)
            .attr('stroke', d => {
                if (d.isDefector) return 'var(--color-defection)';
                return this.getTeamStroke(d.team);
            });

        if (this.linkElements) {
            this.linkElements
                .attr('stroke-opacity', d => Math.max(0.08, Math.min(0.6, d.reciprocity * 0.8)))
                .attr('stroke-width', d => Math.max(0.5, Math.min(3, d.weight * 0.3)));
        }
    },

    updateBadge(count) {
        document.getElementById('topology-badge').textContent =
            `${count} node${count !== 1 ? 's' : ''}`;
    },

    resize() {
        const container = document.getElementById('topology-container');
        this.width = container.clientWidth;
        this.height = container.clientHeight;

        if (this.svg) {
            this.svg
                .attr('width', this.width)
                .attr('height', this.height);
        }

        if (this.simulation) {
            this.simulation
                .force('center', d3.forceCenter(this.width / 2, this.height / 2))
                .force('x', d3.forceX(this.width / 2).strength(0.03))
                .force('y', d3.forceY(this.height / 2).strength(0.03))
                .alpha(0.3)
                .restart();
        }
    },

    reset() {
        if (this.svg) this.svg.selectAll('.links-layer *, .nodes-layer *, .labels-layer *').remove();
        this.nodes = [];
        this.links = [];
        if (this.simulation) this.simulation.nodes([]);
        this.updateBadge(0);
    },
};
