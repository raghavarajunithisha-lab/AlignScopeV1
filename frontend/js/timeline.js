/**
 * AlignScope — Defection & Anomaly Timeline (Dynamic)
 *
 * Horizontal timeline synced to the simulation tick.
 *
 * KEY CHANGE vs original: event type rows are built DYNAMICALLY
 * from whatever event types actually arrive in the data — no
 * hardcoded list. Works with any environment automatically:
 *
 *   KAZ          → only shows "Coalition ↓" row (only type that fires)
 *   Competitive  → shows "Defection" + "Coalition ↓"
 *   Custom env   → shows whatever your detector emits
 *
 * The left-side margin auto-expands to fit the longest label.
 * Marker size = severity. Click/hover shows event details.
 */

const TimelineView = {
    canvas: null,
    ctx: null,
    config: null,
    events: [],
    currentTick: 0,
    maxTicks: 500,
    hoveredEvent: null,
    width: 0,
    height: 0,
    dpr: 1,

    // Visual constants — left margin is computed dynamically
    MARGIN: { top: 16, bottom: 28, right: 20 },
    TRACK_SPACING: 22,   // px between track rows
    TRACK_PADDING: 12,   // px above first track

    // ------------------------------------------------------------------ //
    // DYNAMIC type registry — built from incoming event data              //
    // Keys are event.type strings, values are assigned on first sight     //
    // ------------------------------------------------------------------ //

    // Default color palette — cycles for unknown types
    _colorPalette: [
        '#dc4a4a',  // red      — defection
        '#d4a843',  // amber    — coalition
        '#b06ec7',  // purple   — stability
        '#c97b3a',  // orange   — reciprocity
        '#4abe7d',  // green
        '#6d9eeb',  // blue
        '#e8925a',  // coral
        '#5bc0de',  // teal
        '#8cc152',  // lime
        '#f06292',  // pink
    ],

    // Preferred colors for known type names (override palette cycling)
    _knownColors: {
        defection: '#dc4a4a',
        reciprocity_drop: '#c97b3a',
        stability_drop: '#b06ec7',
        coalition_fragmentation: '#d4a843',
    },

    // Human-readable labels for known type names
    _knownLabels: {
        defection: 'Defection',
        reciprocity_drop: 'Reciprocity ↓',
        stability_drop: 'Stability ↓',
        coalition_fragmentation: 'Coalition ↓',
    },

    // Dynamically built: { type_string: { color, label, index } }
    _typeRegistry: {},

    // ------------------------------------------------------------------ //
    // Helpers                                                              //
    // ------------------------------------------------------------------ //

    /**
     * Register a new event type if not yet seen.
     * Returns the registry entry for this type.
     */
    _registerType(type) {
        if (this._typeRegistry[type]) return this._typeRegistry[type];

        const index = Object.keys(this._typeRegistry).length;
        const color = this._knownColors[type]
            || this._colorPalette[index % this._colorPalette.length];

        // Convert snake_case to "Title ↓" if not a known label
        const label = this._knownLabels[type] || this._formatLabel(type);

        this._typeRegistry[type] = { color, label, index };
        return this._typeRegistry[type];
    },

    _formatLabel(type) {
        // snake_case → "Snake case"
        return type
            .replace(/_/g, ' ')
            .replace(/\b\w/g, c => c.toUpperCase());
    },

    /**
     * Compute the left margin needed to fit the longest label.
     * Recalculated whenever new types are registered.
     */
    _computeLeftMargin() {
        const labels = Object.values(this._typeRegistry).map(t => t.label);
        if (labels.length === 0) return 60;
        // Approximate: ~6.5px per character at 10px Inter
        const maxLen = Math.max(...labels.map(l => l.length));
        return Math.max(60, maxLen * 6.5 + 14);
    },

    /**
     * Y position for a given type's track row.
     */
    _trackY(typeEntry) {
        const left = this._computeLeftMargin();
        const top = this.MARGIN.top;
        return top + this.TRACK_PADDING + typeEntry.index * this.TRACK_SPACING;
    },

    // ------------------------------------------------------------------ //
    // Lifecycle                                                            //
    // ------------------------------------------------------------------ //

    init(config) {
        this.config = config;
        this.maxTicks = config.max_ticks || 500;
        this._typeRegistry = {};  // reset on new episode

        this.canvas = document.getElementById('timeline-canvas');
        if (!this.canvas) { console.error('timeline-canvas not found'); return; }
        this.ctx = this.canvas.getContext('2d');
        this.dpr = window.devicePixelRatio || 1;
        this.zoomLevel = 1.0;

        this.resize();
        this.draw();

        this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        window.addEventListener('mouseup', (e) => this.handleMouseUp(e));
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.canvas.addEventListener('mouseleave', () => this.hideTooltip());
        this.canvas.addEventListener('wheel', (e) => this.handleWheel(e), { passive: false });
    },

    update(data) {
        // Detect episode resets (tick goes backward)
        if (this.lastSeenTick !== undefined && data.tick < this.lastSeenTick - 50) {
            this.currentEpisode = (this.currentEpisode || 1) + 1;
        }
        this.lastSeenTick = data.tick;
        if (this.currentEpisode === undefined) this.currentEpisode = 1;

        this.currentTick = data.tick;
        
        // Dynamically scale max ticks if the simulation runs longer than initially expected
        if (this.currentTick > this.maxTicks) {
            let scaleStep = 500;
            if (this.maxTicks >= 10000) scaleStep = 5000;
            else if (this.maxTicks >= 2000) scaleStep = 1000;
            
            while (this.maxTicks < this.currentTick) {
                this.maxTicks += scaleStep;
            }
        }

        if (data.events && data.events.length > 0) {
            // Register any new event types before accumulating
            for (const event of data.events) {
                event.episode = this.currentEpisode; // Tag event with its episode
                this._registerType(event.type);
            }
            this.events.push(...data.events);
        }

        this.draw();
        this.updateBadge();
    },

    // ------------------------------------------------------------------ //
    // Drawing                                                              //
    // ------------------------------------------------------------------ //

    draw() {
        if (!this.ctx) return;

        const { ctx, width, height, dpr } = this;
        const left = this._computeLeftMargin();
        const { top, bottom, right } = this.MARGIN;
        const plotWidth = width - left - right;
        const plotHeight = height - top - bottom;

        ctx.clearRect(0, 0, width * dpr, height * dpr);
        ctx.save();
        ctx.scale(dpr, dpr);

        // Background
        ctx.fillStyle = '#161922';
        ctx.fillRect(0, 0, width, height);

        // Axis + grid lines
        this.drawAxis(left, top, plotWidth, plotHeight);

        // Progress line
        const progressX = left + (this.currentTick / this.maxTicks) * plotWidth;
        ctx.strokeStyle = '#3d4160';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(progressX, top);
        ctx.lineTo(progressX, top + plotHeight);
        ctx.stroke();
        ctx.setLineDash([]);

        // Current tick label
        ctx.fillStyle = '#9ca3af';
        ctx.font = '10px "JetBrains Mono"';
        ctx.textAlign = 'center';
        ctx.fillText(`t=${this.currentTick}`, progressX, top + plotHeight + 18);

        const numTypes = Object.keys(this._typeRegistry).length;

        if (numTypes === 0) {
            // No events yet — show a placeholder message
            ctx.fillStyle = '#4b5563';
            ctx.font = '11px Inter';
            ctx.textAlign = 'left';
            ctx.fillText('No events yet — waiting for data…', left + 8, top + plotHeight / 2);
        }

        // Draw track row labels (left side) — only for seen types
        ctx.textAlign = 'right';
        ctx.font = '10px Inter';
        for (const [type, entry] of Object.entries(this._typeRegistry)) {
            const y = this._trackY(entry);
            ctx.fillStyle = entry.color;
            ctx.globalAlpha = 0.75;
            ctx.fillText(entry.label, left - 6, y + 3);
            ctx.globalAlpha = 1;

            // Subtle horizontal guide line for this track
            ctx.strokeStyle = entry.color + '18';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(left, y);
            ctx.lineTo(left + plotWidth, y);
            ctx.stroke();
        }

        // Draw event markers
        for (const event of this.events) {
            const entry = this._typeRegistry[event.type];
            if (!entry) continue;

            const x = left + (event.tick / this.maxTicks) * plotWidth;
            const y = this._trackY(entry);
            const severity = event.severity || 0.5;
            const radius = 3 + severity * 5;
            const color = entry.color;

            // Glow for high severity
            if (severity > 0.6) {
                ctx.beginPath();
                ctx.arc(x, y, radius + 3, 0, Math.PI * 2);
                ctx.fillStyle = color + '20';
                ctx.fill();
            }

            // Marker fill
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.globalAlpha = 0.5 + severity * 0.5;
            ctx.fill();
            ctx.globalAlpha = 1;

            // Marker border
            ctx.strokeStyle = color;
            ctx.lineWidth = 1;
            ctx.stroke();
        }

        // Faded overlay for future ticks
        if (progressX < left + plotWidth) {
            ctx.fillStyle = 'rgba(15, 17, 23, 0.4)';
            ctx.fillRect(progressX, top, left + plotWidth - progressX, plotHeight);
        }

        ctx.restore();
    },

    drawAxis(x, y, plotWidth, plotHeight) {
        const ctx = this.ctx;

        ctx.strokeStyle = '#252836';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, y + plotHeight);
        ctx.lineTo(x + plotWidth, y + plotHeight);
        ctx.stroke();

        ctx.fillStyle = '#4b5563';
        ctx.font = '9px "JetBrains Mono"';
        ctx.textAlign = 'center';

        // Dynamic grid interval based on total timeline length
        let interval = 100;
        if (this.maxTicks >= 10000) interval = 1000;
        else if (this.maxTicks >= 1000) interval = 500;
        else if (this.maxTicks <= 200) interval = 50;

        const plotWidthRatio = plotWidth / this.maxTicks;
        
        // Prevent text overlapping when zoomed out
        while (interval * plotWidthRatio < 30) {
            interval *= 2;
        }
        
        // Show finer details when zoomed in
        while (interval * plotWidthRatio > 120 && interval > 1) {
            if (interval === 500) interval = 100;
            else if (interval === 100) interval = 50;
            else if (interval === 50) interval = 10;
            else if (interval === 10) interval = 5;
            else interval /= 2;
        }

        for (let t = 0; t <= this.maxTicks; t += interval) {
            const tickX = x + (t / this.maxTicks) * plotWidth;

            ctx.strokeStyle = '#1e2130';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(tickX, y);
            ctx.lineTo(tickX, y + plotHeight);
            ctx.stroke();

            ctx.fillStyle = '#4b5563';
            ctx.fillText(t.toString(), tickX, y + plotHeight + 12);
        }
    },

    // ------------------------------------------------------------------ //
    // Interaction                                                          //
    // ------------------------------------------------------------------ //

    handleWheel(e) {
        // Only zoom if shift or ctrl is pressed (otherwise scroll page as normal)
        if (!e.shiftKey && !e.ctrlKey) return;
        
        e.preventDefault();
        
        // Calculate mouse position relative to timeline to maintain focus point
        const container = this.canvas.parentElement;
        const mouseX = e.clientX - container.getBoundingClientRect().left;
        const scrollRatio = (container.scrollLeft + mouseX) / this.width;

        const zoomDelta = e.deltaY > 0 ? -0.3 : 0.3;
        this.zoomLevel = Math.max(1.0, Math.min(30.0, this.zoomLevel + zoomDelta));
        this.resize();
        
        // Restore scroll position so zooming feels centered on mouse
        container.scrollLeft = (scrollRatio * this.width) - mouseX;
    },

    handleMouseDown(e) {
        this.isDragging = true;
        this.dragStartX = e.clientX;
        this.lastX = e.clientX;
        this.canvas.style.cursor = 'grabbing';
        this.hideTooltip();
    },

    handleMouseUp(e) {
        if (!this.isDragging) return;
        this.isDragging = false;
        this.canvas.style.cursor = 'pointer';
        
        // If it was a clean click (no drag), handle it as a click
        if (Math.abs(e.clientX - this.dragStartX) < 3) {
            const event = this.findEventAtPos(e);
            if (event) this.showTooltip(event, e);
        }
    },

    handleMouseMove(e) {
        if (this.isDragging) {
            const dx = e.clientX - this.lastX;
            this.canvas.parentElement.scrollLeft -= dx;
            this.lastX = e.clientX;
            return;
        }

        const event = this.findEventAtPos(e);
        this.canvas.style.cursor = event ? 'pointer' : (this.zoomLevel > 1 ? 'grab' : 'default');

        if (event && event !== this.hoveredEvent) {
            this.hoveredEvent = event;
            this.showTooltip(event, e);
        } else if (!event) {
            this.hoveredEvent = null;
            this.hideTooltip();
        }
    },

    findEventAtPos(mouseEvent) {
        const rect = this.canvas.getBoundingClientRect();
        const mx = mouseEvent.clientX - rect.left;
        const my = mouseEvent.clientY - rect.top;

        const left = this._computeLeftMargin();
        const plotWidth = this.width - left - this.MARGIN.right;

        for (const event of this.events) {
            const entry = this._typeRegistry[event.type];
            if (!entry) continue;

            const x = left + (event.tick / this.maxTicks) * plotWidth;
            const y = this._trackY(entry);
            const radius = 3 + (event.severity || 0.5) * 5 + 4;

            if (Math.hypot(mx - x, my - y) <= radius) return event;
        }
        return null;
    },

    showTooltip(event, mouseEvent) {
        const tooltip = document.getElementById('timeline-tooltip');
        const container = document.getElementById('timeline-container');
        const containerRect = container.getBoundingClientRect();

        tooltip.classList.remove('hidden');

        const entry = this._typeRegistry[event.type] || {};
        const label = entry.label || event.type;
        const color = entry.color || '#e4e4e7';

        const epStr = event.episode ? ` | Ep ${event.episode}` : '';
        const agentStr = event.agent_id ? ` | Agent ${event.agent_id}` : '';

        document.getElementById('tooltip-header').textContent = `${label} at tick ${event.tick}${epStr}${agentStr}`;
        document.getElementById('tooltip-header').style.color = color;
        document.getElementById('tooltip-body').textContent = event.description;

        // Position tooltip firmly within the container so it doesn't get clipped by overflow:hidden parents
        const hoverX = mouseEvent.clientX - containerRect.left;
        const hoverY = mouseEvent.clientY - containerRect.top;
        
        // Offset slightly from cursor so it doesn't block hover events
        tooltip.style.left = Math.min(hoverX + 16, containerRect.width - 250) + 'px';
        tooltip.style.top = Math.max(4, hoverY - tooltip.offsetHeight - 16) + 'px';
        tooltip.style.bottom = 'auto'; // override CSS default bottom: 100%
    },

    hideTooltip() {
        const el = document.getElementById('timeline-tooltip');
        if (el) el.classList.add('hidden');
    },

    updateBadge() {
        const el = document.getElementById('timeline-badge');
        if (el) el.textContent = `${this.events.length} event${this.events.length !== 1 ? 's' : ''}`;
    },

    resize() {
        const container = document.getElementById('timeline-container');
        this.baseWidth = container.clientWidth;
        this.height = container.clientHeight;

        // Apply zoom multiplier
        this.width = Math.max(this.baseWidth, this.baseWidth * (this.zoomLevel || 1.0));

        this.canvas.width = this.width * this.dpr;
        this.canvas.height = this.height * this.dpr;
        this.canvas.style.width = this.width + 'px';
        this.canvas.style.height = this.height + 'px';

        this.draw();
    },

    reset() {
        this.events = [];
        this.currentTick = 0;
        this.lastSeenTick = 0;
        this.currentEpisode = 1;
        this.hoveredEvent = null;
        this._typeRegistry = {};
        this.hideTooltip();
        this.draw();
        this.updateBadge();
    },
};
