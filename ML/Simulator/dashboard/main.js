const strategySelect = document.getElementById("strategy-select");
const seasonSelect = document.getElementById("season-select");
const timescaleSelect = document.getElementById("timescale-select");
const householdFilter = document.getElementById("household-filter");
const wealthFilter = document.getElementById("wealth-filter");
const sortSelect = document.getElementById("sort-select");
const sortOrderSelect = document.getElementById("sort-order-select");

const duckPlayBtn = document.getElementById("duck-play-btn");
const duckResetBtn = document.getElementById("duck-reset-btn");
const duckSlider = document.getElementById("duck-slider");
const duckSliderLabel = document.getElementById("duck-slider-label");

const tooltip = d3.select("body")
    .append("div")
    .attr("class", "tooltip");

const DATA_BASE = "../output/dashboard_data";

let dashboardState = {
    rawDatasets: null,
    animatedDuckData: [],
    duckAnimationIndex: 1,
    duckAnimationInterval: null,
    isDuckPlaying: false,

    duckWindowSize: 48
};

function formatKwh(value) {
    return `${d3.format(",.2f")(value)} kWh`;
}

function formatCurrency(value) {
    return `$${d3.format(",.2f")(value)}`;
}

function formatPercent(value) {
    return `${d3.format(".1%")(value)}`;
}

function clearContainer(selector) {
    d3.select(selector).selectAll("*").remove();
}

function showErrorInContainer(selector, message) {
    clearContainer(selector);

    d3.select(selector)
        .append("div")
        .attr("class", "error-box")
        .text(message);
}

async function loadJson(fileName) {
    const path = `${DATA_BASE}/${fileName}`;
    try {
        return await d3.json(path);
    } catch (error) {
        console.error(`Error loading ${path}`, error);
        return null;
    }
}

function resetKpis() {
    document.getElementById("kpi-generation").textContent = "-";
    document.getElementById("kpi-consumption").textContent = "-";
    document.getElementById("kpi-import").textContent = "-";
    document.getElementById("kpi-export").textContent = "-";
    document.getElementById("kpi-self-consumption").textContent = "-";
    document.getElementById("kpi-self-sufficiency").textContent = "-";
}

function populateFilterOptions(costsData) {
    const currentHouseholdValue = householdFilter.value || "all";
    const currentWealthValue = wealthFilter.value || "all";

    const householdTypes = Array.from(new Set(costsData.map(d => d.household_type))).sort();
    const wealthLevels = Array.from(new Set(costsData.map(d => d.wealth_level))).sort();

    householdFilter.innerHTML = `<option value="all">All</option>`;
    wealthFilter.innerHTML = `<option value="all">All</option>`;

    householdTypes.forEach(type => {
        const option = document.createElement("option");
        option.value = type;
        option.textContent = type;
        householdFilter.appendChild(option);
    });

    wealthLevels.forEach(level => {
        const option = document.createElement("option");
        option.value = level;
        option.textContent = level;
        wealthFilter.appendChild(option);
    });

    if (householdTypes.includes(currentHouseholdValue)) {
        householdFilter.value = currentHouseholdValue;
    } else {
        householdFilter.value = "all";
    }

    if (wealthLevels.includes(currentWealthValue)) {
        wealthFilter.value = currentWealthValue;
    } else {
        wealthFilter.value = "all";
    }
}

function getCurrentFilters() {
    return {
        householdType: householdFilter.value,
        wealthLevel: wealthFilter.value,
        sortMetric: sortSelect.value,
        sortOrder: sortOrderSelect.value,
        timescale: timescaleSelect.value
    };
}

function filterHouseholdRows(data, filters) {
    return data.filter(d => {
        const householdOk = filters.householdType === "all" || d.household_type === filters.householdType;
        const wealthOk = filters.wealthLevel === "all" || d.wealth_level === filters.wealthLevel;
        return householdOk && wealthOk;
    });
}

function aggregateTimeSeries(data, timescale, valueKeys) {
    const grouped = new Map();

    data.forEach(d => {
        const t = +d.time;
        let bucket = t;

        if (timescale === "daily") {
            bucket = Math.floor(t / 24);
        } else if (timescale === "weekly") {
            bucket = Math.floor(t / (24 * 7));
        }

        const current = grouped.get(bucket) || { time: bucket };

        valueKeys.forEach(key => {
            current[key] = (current[key] || 0) + (+d[key] || 0);
        });

        grouped.set(bucket, current);
    });

    return Array.from(grouped.values()).sort((a, b) => a.time - b.time);
}

function updateStory(kpisData, allResultsSummary, season, strategy) {
    const insightsContainer = document.getElementById("insights-list");
    const conclusion = document.getElementById("strategy-conclusion");

    insightsContainer.innerHTML = "";

    const current = kpisData[0];
    const sameSeason = allResultsSummary.filter(d => d.season === season);

    const bestImport = sameSeason.reduce((best, row) =>
        row.total_grid_import_kwh < best.total_grid_import_kwh ? row : best
    );

    const bestSelfConsumption = sameSeason.reduce((best, row) =>
        row.avg_self_consumption_ratio > best.avg_self_consumption_ratio ? row : best
    );

    const lowestCost = sameSeason.reduce((best, row) =>
        row.total_net_cost < best.total_net_cost ? row : best
    );

    const insightTexts = [
        `In ${season}, the lowest grid import comes from ${bestImport.strategy}.`,
        `In ${season}, the highest self-consumption ratio comes from ${bestSelfConsumption.strategy}.`,
        `In ${season}, the lowest total net cost comes from ${lowestCost.strategy}.`
    ];

    insightTexts.forEach(text => {
        const div = document.createElement("div");
        div.className = "insight-item";
        div.textContent = text;
        insightsContainer.appendChild(div);
    });

    conclusion.textContent =
        `${strategy} in ${season} produces ${formatKwh(current.total_generation_kwh)} of total generation, `
        + `${formatKwh(current.total_grid_import_kwh)} of grid import, and an average self-consumption ratio of `
        + `${formatPercent(current.avg_self_consumption_ratio)}. This helps explain whether the strategy favors `
        + `local energy usage, grid dependence, or export-oriented behavior.`;
}

function buildDuckSummaryText(data, timescale) {
    if (!data || data.length === 0) {
        return "No Duck Curve data available.";
    }

    const totalLoad = d3.sum(data, d => +d.load_kwh);
    const totalSolar = d3.sum(data, d => +d.generation_kwh);
    const avgNetLoad = d3.mean(data, d => +d.net_load_kwh);

    const dominantMessage =
        totalSolar > totalLoad * 0.6
            ? "Solar production plays a strong role in offsetting local demand."
            : "Demand remains dominant over solar output for most of the observed period.";

    return `In the selected ${timescale} view, the neighborhood consumes ${d3.format(",.2f")(totalLoad)} kWh and generates ${d3.format(",.2f")(totalSolar)} kWh of solar energy. Average net load is ${d3.format(",.2f")(avgNetLoad)} kWh. ${dominantMessage}`;
}

function buildDuckInsightText(data) {
    if (!data || data.length === 0) {
        return "No key insights available.";
    }

    const peakLoad = data.reduce((best, d) => d.load_kwh > best.load_kwh ? d : best, data[0]);
    const peakSolar = data.reduce((best, d) => d.generation_kwh > best.generation_kwh ? d : best, data[0]);
    const minNetLoad = data.reduce((best, d) => d.net_load_kwh < best.net_load_kwh ? d : best, data[0]);

    return `Peak demand occurs at bucket ${peakLoad.time} (${d3.format(",.2f")(peakLoad.load_kwh)} kWh), peak solar occurs at bucket ${peakSolar.time} (${d3.format(",.2f")(peakSolar.generation_kwh)} kWh), and the lowest net load appears at bucket ${minNetLoad.time} (${d3.format(",.2f")(minNetLoad.net_load_kwh)} kWh), which indicates the strongest local solar relief period.`;
}

function buildTimeseriesSummaryText(data, timescale) {
    if (!data || data.length === 0) {
        return "No summary available for household-type timeseries.";
    }

    const aggregatedByType = [];

    const grouped = d3.group(data, d => d.household_type);

    grouped.forEach((rows, type) => {
        const agg = aggregateTimeSeries(rows, timescale, ["load_kwh"]);

        agg.forEach(row => {
            aggregatedByType.push({
                time: row.time,
                household_type: type,
                load_kwh: +row.load_kwh
            });
        });
    });

    const regrouped = d3.group(aggregatedByType, d => d.household_type);

    const summaries = Array.from(regrouped, ([type, rows]) => {
        const total = d3.sum(rows, d => d.load_kwh);
        const avg = d3.mean(rows, d => d.load_kwh);
        const peak = d3.max(rows, d => d.load_kwh);

        return {
            household_type: type,
            total_load_kwh: total,
            avg_load_kwh: avg,
            peak_load_kwh: peak
        };
    });

    const highestAverage = summaries.reduce((best, row) =>
        row.avg_load_kwh > best.avg_load_kwh ? row : best
    );

    const highestPeak = summaries.reduce((best, row) =>
        row.peak_load_kwh > best.peak_load_kwh ? row : best
    );

    const lowestAverage = summaries.reduce((best, row) =>
        row.avg_load_kwh < best.avg_load_kwh ? row : best
    );

    return `${highestAverage.household_type} shows the highest average consumption at ${d3.format(",.2f")(highestAverage.avg_load_kwh)} kWh per ${timescale} bucket. `
        + `${highestPeak.household_type} reaches the highest peak at ${d3.format(",.2f")(highestPeak.peak_load_kwh)} kWh. `
        + `${lowestAverage.household_type} remains the lowest-average consumer at ${d3.format(",.2f")(lowestAverage.avg_load_kwh)} kWh.`;
}

function formatDuckSliderLabel(value, max, timescale) {
    const windowSize = Math.min(dashboardState.duckWindowSize, max);
    const endValue = +value;
    const startValue = Math.max(1, endValue - windowSize + 1);

    if (timescale === "hourly") {
        return `Showing hours ${startValue}–${endValue}`;
    } else if (timescale === "daily") {
        return `Showing days ${startValue}–${endValue}`;
    } else if (timescale === "weekly") {
        return `Showing weeks ${startValue}–${endValue}`;
    } else {
        return `Showing ${startValue}–${endValue}`;
    }
}

function getDuckMovingWindow(data) {
    if (!data || data.length === 0) {
        return [];
    }

    const windowSize = Math.min(dashboardState.duckWindowSize, data.length);
    const endIndex = Math.max(windowSize, dashboardState.duckAnimationIndex);
    const startIndex = Math.max(0, endIndex - windowSize);

    return data.slice(startIndex, endIndex);
}

function setupDuckAnimation(data) {
    stopDuckAnimation();

    dashboardState.animatedDuckData = data;

    dashboardState.duckAnimationIndex = Math.min(dashboardState.duckWindowSize,data.length || 1);
    dashboardState.isDuckPlaying = false;

    duckPlayBtn.textContent = "Play";
    duckSlider.min = 1;
    duckSlider.max = Math.max(1, data.length);
    duckSlider.value = dashboardState.duckAnimationIndex;
    duckSliderLabel.textContent = formatDuckSliderLabel(
        duckSlider.value,
        duckSlider.max,
        getCurrentFilters().timescale
    );
}

function stopDuckAnimation() {
    if (dashboardState.duckAnimationInterval) {
        clearInterval(dashboardState.duckAnimationInterval);
        dashboardState.duckAnimationInterval = null;
    }
    dashboardState.isDuckPlaying = false;
    duckPlayBtn.textContent = "Play";
}

function playDuckAnimation() {
    if (!dashboardState.animatedDuckData || dashboardState.animatedDuckData.length === 0) {
        return;
    }

    if (dashboardState.isDuckPlaying) {
        stopDuckAnimation();
        return;
    }

    dashboardState.isDuckPlaying = true;
    duckPlayBtn.textContent = "Pause";

    dashboardState.duckAnimationInterval = setInterval(() => {
        const maxSteps = dashboardState.animatedDuckData.length;

        if (dashboardState.duckAnimationIndex >= maxSteps) {
            stopDuckAnimation();
            return;
        }

        dashboardState.duckAnimationIndex += 1;
        duckSlider.value = dashboardState.duckAnimationIndex;
        duckSliderLabel.textContent = formatDuckSliderLabel(
            duckSlider.value,
            duckSlider.max,
            getCurrentFilters().timescale
        );

        renderAllCharts(dashboardState.rawDatasets);
    }, 120);
}

function resetDuckAnimation() {
    stopDuckAnimation();

    if (!dashboardState.animatedDuckData || dashboardState.animatedDuckData.length === 0) {
        return;
    }

    dashboardState.duckAnimationIndex = Math.min(
        dashboardState.duckWindowSize,
        dashboardState.animatedDuckData.length || 1
    );

duckSlider.value = dashboardState.duckAnimationIndex;

    duckSliderLabel.textContent = formatDuckSliderLabel(
        duckSlider.value,
        duckSlider.max,
        getCurrentFilters().timescale
    );

    renderAllCharts(dashboardState.rawDatasets);
}

async function loadDashboard() {
    const season = seasonSelect.value;
    const strategy = strategySelect.value;

    stopDuckAnimation();

    const [
        kpis,
        byHouseholdType,
        byWealthLevel,
        duckCurve,
        costsSelfConsumption,
        timeseriesByType,
        allResultsSummary,

        mlMetrics,
        mlPredictions,
        mlResiduals,
        idealVsML,
        forecastComparison

    ] = await Promise.all([
        loadJson(`kpis_${season}_${strategy}.json`),
        loadJson(`by_household_type_${season}_${strategy}.json`),
        loadJson(`by_wealth_level_${season}_${strategy}.json`),
        loadJson(`duck_curve_${season}_${strategy}.json`),
        loadJson(`costs_self_consumption_${season}_${strategy}.json`),
        loadJson(`timeseries_by_household_type_${season}_${strategy}.json`),
        loadJson(`all_results_summary.json`),

        loadJson(`ml_model_metrics.json`),
        loadJson(`ml_prediction_sample.json`),
        loadJson(`ml_residual_error.json`),
        loadJson(`ideal_vs_ml_solar.json`),
        loadJson(`forecast_comparison.json`)
    ]);

    if (!kpis || !byHouseholdType || !byWealthLevel || !duckCurve || !costsSelfConsumption || !timeseriesByType || !allResultsSummary) {
        resetKpis();

        showErrorInContainer(
            "#duck-curve-chart",
            `No data found for Season=${season} and Strategy=${strategy}. Run greengridsim.py again so the missing JSON files are generated.`
        );
        showErrorInContainer("#household-type-chart", "Missing dataset.");
        showErrorInContainer("#wealth-level-chart", "Missing dataset.");
        showErrorInContainer("#costs-scatter-chart", "Missing dataset.");
        showErrorInContainer("#timeseries-household-type-chart", "Missing dataset.");
        showErrorInContainer("#battery-utilization-chart", "Missing dataset.");
        showErrorInContainer("#surplus-deficit-chart", "Missing dataset.");
        return;
    }

    populateFilterOptions(costsSelfConsumption);
    updateKpis(kpis[0]);
    updateStory(kpis, allResultsSummary, season, strategy);

    dashboardState.rawDatasets = {
        byHouseholdType,
        byWealthLevel,
        duckCurve,
        costsSelfConsumption,
        timeseriesByType,
        mlMetrics,
        mlPredictions,
        mlResiduals,
        idealVsML,
        forecastComparison
    };

    const filters = getCurrentFilters();
    const aggregatedDuck = aggregateTimeSeries(
        duckCurve,
        filters.timescale,
        ["generation_kwh", "load_kwh", "self_consumed_kwh", "grid_import", "grid_export", "net_load_kwh"]
    );

    setupDuckAnimation(aggregatedDuck);
    updateMLMetrics(mlMetrics);
    updateExecutiveSummary(kpis, allResultsSummary, mlMetrics,forecastComparison);
    renderAllCharts(dashboardState.rawDatasets);
}

function renderAllCharts(datasets) {
    if (!datasets) return;

    const filters = getCurrentFilters();

    const filteredHouseholds = filterHouseholdRows(datasets.costsSelfConsumption, filters);

    const filteredTimeseries = datasets.timeseriesByType.filter(d => {
        const typeOk = filters.householdType === "all" || d.household_type === filters.householdType;
        return typeOk;
    });

    const aggregatedDuck = aggregateTimeSeries(
        datasets.duckCurve,
        filters.timescale,
        ["generation_kwh", "load_kwh", "self_consumed_kwh", "grid_import", "grid_export", "net_load_kwh"]
    );

    dashboardState.animatedDuckData = aggregatedDuck;

    duckSlider.max = Math.max(1, aggregatedDuck.length);
    if (dashboardState.duckAnimationIndex > aggregatedDuck.length) {
        dashboardState.duckAnimationIndex = aggregatedDuck.length;
    }
    if (dashboardState.duckAnimationIndex < 1) {
        dashboardState.duckAnimationIndex = 1;
    }

    duckSlider.value = dashboardState.duckAnimationIndex;
    duckSliderLabel.textContent = formatDuckSliderLabel(
        duckSlider.value,
        duckSlider.max,
        getCurrentFilters().timescale
    );

    const partialDuck = getDuckMovingWindow(aggregatedDuck);

    document.getElementById("duck-summary-text").textContent = buildDuckSummaryText(aggregatedDuck, filters.timescale);
    document.getElementById("duck-insight-text").textContent = buildDuckInsightText(aggregatedDuck);

    drawDuckCurve("#duck-curve-chart", partialDuck, filters.timescale);
    drawHouseholdTypeBar("#household-type-chart", datasets.byHouseholdType, filters);
    drawWealthGroupedBar("#wealth-level-chart", datasets.byWealthLevel, filters);
    drawCostsScatter("#costs-scatter-chart", filteredHouseholds);
    
    drawTimeseriesByType("#timeseries-household-type-chart", filteredTimeseries, filters.timescale);

    document.getElementById("timeseries-summary-text").textContent =
    buildTimeseriesSummaryText(filteredTimeseries, filters.timescale);

    drawBatteryUtilization("#battery-utilization-chart", datasets.byHouseholdType, filters);
    drawSurplusDeficit("#surplus-deficit-chart", aggregatedDuck, filters.timescale);
    drawMLPredictionChart("#ml-prediction-chart",datasets.mlPredictions);
    drawMLResidualChart("#ml-residual-chart",datasets.mlResiduals);
    drawIdealVsMLChart("#ideal-vs-ml-chart",datasets.idealVsML);
    drawForecastComparisonChart("#forecast-comparison-chart",datasets.forecastComparison);
}

function updateKpis(data) {
    document.getElementById("kpi-generation").textContent = formatKwh(data.total_generation_kwh);
    document.getElementById("kpi-consumption").textContent = formatKwh(data.total_consumption_kwh);
    document.getElementById("kpi-import").textContent = formatKwh(data.total_grid_import_kwh);
    document.getElementById("kpi-export").textContent = formatKwh(data.total_grid_export_kwh);
    document.getElementById("kpi-self-consumption").textContent = formatPercent(data.avg_self_consumption_ratio);
    document.getElementById("kpi-self-sufficiency").textContent = formatPercent(data.avg_self_sufficiency_ratio);
}

function createSvg(containerSelector, width = 1000, height = 420, margin = { top: 30, right: 30, bottom: 60, left: 70 }) {
    clearContainer(containerSelector);

    const svg = d3.select(containerSelector)
        .append("svg")
        .attr("viewBox", `0 0 ${width} ${height}`);

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg.append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    return { svg, g, width, height, innerWidth, innerHeight, margin };
}

function drawDuckCurve(containerSelector, data, timescaleLabel) {
    if (!data || data.length === 0) {
        showErrorInContainer(containerSelector, "No data to display in Duck Curve.");
        return;
    }

    data.forEach(d => {
        d.time = +d.time;
        d.generation_kwh = +d.generation_kwh;
        d.load_kwh = +d.load_kwh;
        d.net_load_kwh = +d.net_load_kwh;
    });

    const { svg, g, innerWidth, innerHeight } = createSvg(
        containerSelector,
        1250,
        500,
        { top: 40, right: 40, bottom: 80, left: 85 }
    );

    const minX = d3.min(data, d => d.time);
    const maxX = d3.max(data, d => d.time);

    const maxY = d3.max(data, d =>
        Math.max(d.load_kwh, d.generation_kwh, d.net_load_kwh)
    ) || 1;

    const minY = d3.min(data, d => d.net_load_kwh) || 0;

    const x = d3.scaleLinear()
        .domain([minX, maxX])
        .range([0, innerWidth]);

    const y = d3.scaleLinear()
        .domain([
            Math.min(0, minY * 1.1),
            maxY * 1.15
        ])
        .range([innerHeight, 0]);

    const xTicks =
        timescaleLabel === "hourly" ? 12 :
        timescaleLabel === "daily" ? 10 :
        8;

    const yTicks = 8;

    const xAxis = d3.axisBottom(x)
        .ticks(xTicks)
        .tickFormat(d3.format("d"));

    const yAxis = d3.axisLeft(y)
        .ticks(yTicks)
        .tickFormat(d3.format(".1f"));

    g.append("g")
        .attr("class", "grid")
        .attr("transform", `translate(0,${innerHeight})`)
        .call(
            d3.axisBottom(x)
                .ticks(xTicks)
                .tickSize(-innerHeight)
                .tickFormat("")
        );

    g.append("g")
        .attr("class", "grid")
        .call(
            d3.axisLeft(y)
                .ticks(yTicks)
                .tickSize(-innerWidth)
                .tickFormat("")
        );

    g.append("g")
        .attr("class", "axis")
        .attr("transform", `translate(0,${innerHeight})`)
        .call(xAxis);

    g.append("g")
        .attr("class", "axis")
        .call(yAxis);

    const series = [
        { key: "load_kwh", color: "#38bdf8", label: "Load", strokeWidth: 2.5 },
        { key: "generation_kwh", color: "#22c55e", label: "Solar", strokeWidth: 2.5 },
        { key: "net_load_kwh", color: "#f59e0b", label: "Net Load", strokeWidth: 3.2 }
    ];

    const lineGenerator = key => d3.line()
        .x(d => x(d.time))
        .y(d => y(d[key]));

    const linePaths = {};

    series.forEach(s => {
        linePaths[s.key] = g.append("path")
            .datum(data)
            .attr("fill", "none")
            .attr("stroke", s.color)
            .attr("stroke-width", s.strokeWidth)
            .attr("opacity", 0.95)
            .attr("d", lineGenerator(s.key))
            .on("mouseenter", function () {
                Object.values(linePaths).forEach(path => path.attr("opacity", 0.18));
                linePaths[s.key].attr("opacity", 1).attr("stroke-width", s.strokeWidth + 1.4);
            })
            .on("mouseleave", function () {
                Object.values(linePaths).forEach(path => path.attr("opacity", 0.95));
                linePaths[s.key].attr("stroke-width", s.strokeWidth);
            });
    });

    const peakLoad = data.reduce((best, d) => d.load_kwh > best.load_kwh ? d : best, data[0]);
    const peakSolar = data.reduce((best, d) => d.generation_kwh > best.generation_kwh ? d : best, data[0]);
    const minNetLoad = data.reduce((best, d) => d.net_load_kwh < best.net_load_kwh ? d : best, data[0]);

    const keyMoments = [
        {
            x: peakLoad.time,
            y: peakLoad.load_kwh,
            color: "#38bdf8",
            label: "Peak Load",
            valueText: `${d3.format(",.2f")(peakLoad.load_kwh)} kWh`
        },
        {
            x: peakSolar.time,
            y: peakSolar.generation_kwh,
            color: "#22c55e",
            label: "Peak Solar",
            valueText: `${d3.format(",.2f")(peakSolar.generation_kwh)} kWh`
        },
        {
            x: minNetLoad.time,
            y: minNetLoad.net_load_kwh,
            color: "#f59e0b",
            label: "Minimum Net Load",
            valueText: `${d3.format(",.2f")(minNetLoad.net_load_kwh)} kWh`
        }
    ];

    keyMoments.forEach(moment => {
        g.append("line")
            .attr("x1", x(moment.x))
            .attr("x2", x(moment.x))
            .attr("y1", 0)
            .attr("y2", innerHeight)
            .attr("stroke", moment.color)
            .attr("stroke-width", 1.2)
            .attr("stroke-dasharray", "4,4")
            .attr("opacity", 0.35);

        g.append("circle")
            .attr("cx", x(moment.x))
            .attr("cy", y(moment.y))
            .attr("r", 6)
            .attr("fill", moment.color)
            .attr("stroke", "#e5e7eb")
            .attr("stroke-width", 1.2)
            .on("mousemove", (event) => {
                tooltip
                    .style("opacity", 1)
                    .html(`
                        <strong>${moment.label}</strong><br>
                        Bucket: ${moment.x}<br>
                        Value: ${moment.valueText}
                    `)
                    .style("left", `${event.pageX + 12}px`)
                    .style("top", `${event.pageY - 28}px`);
            })
            .on("mouseleave", () => tooltip.style("opacity", 0));
    });

    const overlay = g.append("rect")
        .attr("width", innerWidth)
        .attr("height", innerHeight)
        .attr("fill", "transparent");

    const focusLine = g.append("line")
        .attr("stroke", "#94a3b8")
        .attr("stroke-width", 1.2)
        .attr("stroke-dasharray", "4,4")
        .attr("y1", 0)
        .attr("y2", innerHeight)
        .style("opacity", 0);

    const focusLoad = g.append("circle")
        .attr("r", 5)
        .attr("fill", "#38bdf8")
        .style("opacity", 0);

    const focusSolar = g.append("circle")
        .attr("r", 5)
        .attr("fill", "#22c55e")
        .style("opacity", 0);

    const focusNet = g.append("circle")
        .attr("r", 5)
        .attr("fill", "#f59e0b")
        .style("opacity", 0);

    overlay
        .on("mousemove", function (event) {
            const [mx] = d3.pointer(event, this);
            const hoveredTime = x.invert(mx);

            let closest = data[0];
            let minDistance = Math.abs(data[0].time - hoveredTime);

            for (let i = 1; i < data.length; i++) {
                const distance = Math.abs(data[i].time - hoveredTime);
                if (distance < minDistance) {
                    minDistance = distance;
                    closest = data[i];
                }
            }

            focusLine
                .style("opacity", 1)
                .attr("x1", x(closest.time))
                .attr("x2", x(closest.time));

            focusLoad
                .style("opacity", 1)
                .attr("cx", x(closest.time))
                .attr("cy", y(closest.load_kwh));

            focusSolar
                .style("opacity", 1)
                .attr("cx", x(closest.time))
                .attr("cy", y(closest.generation_kwh));

            focusNet
                .style("opacity", 1)
                .attr("cx", x(closest.time))
                .attr("cy", y(closest.net_load_kwh));

            tooltip
                .style("opacity", 1)
                .html(`
                    <strong>Bucket ${closest.time}</strong><br>
                    Load: ${d3.format(",.2f")(closest.load_kwh)} kWh<br>
                    Solar: ${d3.format(",.2f")(closest.generation_kwh)} kWh<br>
                    Net Load: ${d3.format(",.2f")(closest.net_load_kwh)} kWh
                `)
                .style("left", `${event.pageX + 16}px`)
                .style("top", `${event.pageY - 30}px`);
        })
        .on("mouseleave", function () {
            focusLine.style("opacity", 0);
            focusLoad.style("opacity", 0);
            focusSolar.style("opacity", 0);
            focusNet.style("opacity", 0);
            tooltip.style("opacity", 0);
            Object.values(linePaths).forEach(path => path.attr("opacity", 0.95));
            series.forEach(s => linePaths[s.key].attr("stroke-width", s.strokeWidth));
        });

    const legendData = [
        { label: "Load", color: "#38bdf8" },
        { label: "Solar", color: "#22c55e" },
        { label: "Net Load", color: "#f59e0b" }
    ];

    const legend = g.selectAll(".duck-legend")
        .data(legendData)
        .enter()
        .append("g")
        .attr("transform", (d, i) => `translate(${innerWidth - 150}, ${10 + i * 24})`);

    legend.append("rect")
        .attr("width", 14)
        .attr("height", 14)
        .attr("fill", d => d.color);

    legend.append("text")
        .attr("x", 22)
        .attr("y", 12)
        .attr("class", "legend")
        .text(d => d.label);

    g.append("text")
        .attr("x", innerWidth)
        .attr("y", -15)
        .attr("text-anchor", "end")
        .attr("fill", "#94a3b8")
        .attr("font-size", 12)
        .text(`Grouped by: ${timescaleLabel}`);

    g.append("text")
        .attr("x", innerWidth / 2)
        .attr("y", innerHeight + 55)
        .attr("text-anchor", "middle")
        .attr("fill", "#d1d5db")
        .attr("font-size", 14)
        .text(
            timescaleLabel === "hourly"
                ? "Time Bucket (Hours)"
                : timescaleLabel === "daily"
                ? "Time Bucket (Days)"
                : "Time Bucket (Weeks)"
        );

    g.append("text")
        .attr("transform", "rotate(-90)")
        .attr("x", -innerHeight / 2)
        .attr("y", -55)
        .attr("text-anchor", "middle")
        .attr("fill", "#d1d5db")
        .attr("font-size", 14)
        .text("Energy (kWh)");
}

function drawHouseholdTypeBar(containerSelector, data, filters) {
    const metric = filters.sortMetric;
    const sortOrder = filters.sortOrder;

    const filtered = data
        .filter(d => filters.householdType === "all" || d.household_type === filters.householdType)
        .sort((a, b) => {
            const av = +a[metric];
            const bv = +b[metric];
            return sortOrder === "asc" ? av - bv : bv - av;
        });

    const { g, innerWidth, innerHeight } = createSvg(containerSelector, 700, 420);

    const x = d3.scaleBand()
        .domain(filtered.map(d => d.household_type))
        .range([0, innerWidth])
        .padding(0.25);

    const y = d3.scaleLinear()
        .domain([0, d3.max(filtered, d => +d[metric]) * 1.1 || 1])
        .range([innerHeight, 0]);

    g.append("g")
        .attr("class", "axis")
        .attr("transform", `translate(0,${innerHeight})`)
        .call(d3.axisBottom(x));

    g.append("g")
        .attr("class", "axis")
        .call(d3.axisLeft(y));

    g.selectAll("rect")
        .data(filtered)
        .enter()
        .append("rect")
        .attr("x", d => x(d.household_type))
        .attr("y", d => y(+d[metric]))
        .attr("width", x.bandwidth())
        .attr("height", d => innerHeight - y(+d[metric]))
        .attr("fill", "#60a5fa")
        .on("mousemove", (event, d) => {
            tooltip
                .style("opacity", 1)
                .html(`
                    <strong>${d.household_type}</strong><br>
                    ${metric}: ${d3.format(",.2f")(d[metric])}
                `)
                .style("left", `${event.pageX + 12}px`)
                .style("top", `${event.pageY - 28}px`);
        })
        .on("mouseleave", () => tooltip.style("opacity", 0));
}

function drawWealthGroupedBar(containerSelector, data, filters) {
    const filtered = data.filter(d => filters.wealthLevel === "all" || d.wealth_level === filters.wealthLevel);

    const { g, innerWidth, innerHeight } = createSvg(containerSelector, 700, 420);

    const groups = filtered.map(d => d.wealth_level);
    const subgroups = ["generation_kwh", "consumption_kwh"];

    const x0 = d3.scaleBand()
        .domain(groups)
        .range([0, innerWidth])
        .padding(0.25);

    const x1 = d3.scaleBand()
        .domain(subgroups)
        .range([0, x0.bandwidth()])
        .padding(0.12);

    const y = d3.scaleLinear()
        .domain([0, d3.max(filtered, d => Math.max(+d.generation_kwh, +d.consumption_kwh)) * 1.1 || 1])
        .range([innerHeight, 0]);

    const color = d3.scaleOrdinal()
        .domain(subgroups)
        .range(["#22c55e", "#38bdf8"]);

    g.append("g")
        .attr("class", "axis")
        .attr("transform", `translate(0,${innerHeight})`)
        .call(d3.axisBottom(x0));

    g.append("g")
        .attr("class", "axis")
        .call(d3.axisLeft(y));

    g.append("g")
        .selectAll("g")
        .data(filtered)
        .enter()
        .append("g")
        .attr("transform", d => `translate(${x0(d.wealth_level)},0)`)
        .selectAll("rect")
        .data(d => subgroups.map(key => ({ key, value: +d[key], wealth_level: d.wealth_level })))
        .enter()
        .append("rect")
        .attr("x", d => x1(d.key))
        .attr("y", d => y(d.value))
        .attr("width", x1.bandwidth())
        .attr("height", d => innerHeight - y(d.value))
        .attr("fill", d => color(d.key))
        .on("mousemove", (event, d) => {
            tooltip
                .style("opacity", 1)
                .html(`
                    <strong>${d.wealth_level}</strong><br>
                    ${d.key}: ${formatKwh(d.value)}
                `)
                .style("left", `${event.pageX + 12}px`)
                .style("top", `${event.pageY - 28}px`);
        })
        .on("mouseleave", () => tooltip.style("opacity", 0));
}

function drawCostsScatter(containerSelector, data) {
    if (!data || data.length === 0) {
        showErrorInContainer(containerSelector, "No data available for Household Scatter.");
        return;
    }

    data.forEach(d => {
        d.household_id = +d.household_id;
        d.net_cost = +d.net_cost;
        d.self_consumption_ratio = +d.self_consumption_ratio;
        d.consumption_kwh = +d.consumption_kwh;
        d.generation_kwh = +d.generation_kwh;
        d.self_consumed_kwh = +d.self_consumed_kwh;
    });

    const filters = getCurrentFilters();

    const { g, innerWidth, innerHeight } = createSvg(
        containerSelector,
        1200,
        470,
        { top: 30, right: 40, bottom: 70, left: 80 }
    );

    const x = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.self_consumption_ratio) * 1.08 || 1])
        .range([0, innerWidth]);

    const maxCost = d3.max(data, d => d.net_cost) ?? 1;
    const minCost = d3.min(data, d => d.net_cost) ?? 0;

    const y = d3.scaleLinear()
        .domain([
            Math.min(0, minCost * 1.12),
            Math.max(0, maxCost * 1.12)
        ])
        .range([innerHeight, 0]);

    const bubbleSize = d3.scaleSqrt()
        .domain(d3.extent(data, d => d.consumption_kwh))
        .range([6, 22]);

    const color = d3.scaleOrdinal()
        .domain(["low", "middle", "high", "luxury"])
        .range(["#60a5fa", "#34d399", "#f59e0b", "#f472b6"]);

    g.append("g")
        .attr("class", "axis")
        .attr("transform", `translate(0,${innerHeight})`)
        .call(d3.axisBottom(x).tickFormat(d3.format(".0%")));

    g.append("g")
        .attr("class", "axis")
        .call(d3.axisLeft(y));

    g.append("text")
        .attr("x", innerWidth / 2)
        .attr("y", innerHeight + 50)
        .attr("text-anchor", "middle")
        .attr("fill", "#d1d5db")
        .attr("font-size", 14)
        .text("Self-Consumption Ratio");

    g.append("text")
        .attr("transform", "rotate(-90)")
        .attr("x", -innerHeight / 2)
        .attr("y", -55)
        .attr("text-anchor", "middle")
        .attr("fill", "#d1d5db")
        .attr("font-size", 14)
        .text("Net Cost ($)");

    const avgX = d3.mean(data, d => d.self_consumption_ratio);
    const avgY = d3.mean(data, d => d.net_cost);

    g.append("line")
        .attr("class", "scatter-reference-line")
        .attr("x1", x(avgX))
        .attr("x2", x(avgX))
        .attr("y1", 0)
        .attr("y2", innerHeight);

    g.append("line")
        .attr("class", "scatter-reference-line")
        .attr("x1", 0)
        .attr("x2", innerWidth)
        .attr("y1", y(avgY))
        .attr("y2", y(avgY));

    g.append("text")
        .attr("class", "scatter-quadrant-label")
        .attr("x", x(avgX) + 12)
        .attr("y", 20)
        .text("High self-consumption / High cost");

    g.append("text")
        .attr("class", "scatter-quadrant-label")
        .attr("x", 12)
        .attr("y", 20)
        .text("Low self-consumption / High cost");

    g.append("text")
        .attr("class", "scatter-quadrant-label")
        .attr("x", 12)
        .attr("y", innerHeight - 10)
        .text("Low self-consumption / Low cost");

    g.append("text")
        .attr("class", "scatter-quadrant-label")
        .attr("x", x(avgX) + 12)
        .attr("y", innerHeight - 10)
        .text("High self-consumption / Low cost");

    const wealthCentroids = Array.from(
        d3.group(data, d => d.wealth_level),
        ([wealth, rows]) => ({
            wealth_level: wealth,
            avg_self_consumption_ratio: d3.mean(rows, d => d.self_consumption_ratio),
            avg_net_cost: d3.mean(rows, d => d.net_cost)
        })
    );

    g.selectAll(".centroid")
        .data(wealthCentroids)
        .enter()
        .append("circle")
        .attr("cx", d => x(d.avg_self_consumption_ratio))
        .attr("cy", d => y(d.avg_net_cost))
        .attr("r", 9)
        .attr("fill", d => color(d.wealth_level))
        .attr("stroke", "#f8fafc")
        .attr("stroke-width", 2);

    g.selectAll(".centroid-label")
        .data(wealthCentroids)
        .enter()
        .append("text")
        .attr("class", "scatter-center-label")
        .attr("x", d => x(d.avg_self_consumption_ratio) + 12)
        .attr("y", d => y(d.avg_net_cost) - 10)
        .text(d => `${d.wealth_level} avg`);

    const circles = g.selectAll(".scatter-bubble")
        .data(data)
        .enter()
        .append("circle")
        .attr("class", "scatter-bubble")
        .attr("cx", d => x(d.self_consumption_ratio))
        .attr("cy", d => y(d.net_cost))
        .attr("r", d => bubbleSize(d.consumption_kwh))
        .attr("fill", d => color(d.wealth_level))
        .attr("stroke", "#e5e7eb")
        .attr("stroke-width", 1.2);

    circles
        .attr("opacity", d => {
            const householdMatch = filters.householdType === "all" || d.household_type === filters.householdType;
            const wealthMatch = filters.wealthLevel === "all" || d.wealth_level === filters.wealthLevel;
            return householdMatch && wealthMatch ? 0.9 : 0.12;
        });

    circles
        .on("mousemove", (event, d) => {
            circles
                .attr("opacity", other => {
                    const householdMatch = filters.householdType === "all" || other.household_type === filters.householdType;
                    const wealthMatch = filters.wealthLevel === "all" || other.wealth_level === filters.wealthLevel;
                    return householdMatch && wealthMatch ? 0.18 : 0.08;
                })
                .attr("stroke-width", 1.2);

            d3.select(event.currentTarget)
                .attr("opacity", 1)
                .attr("stroke-width", 2.5);

            tooltip
                .style("opacity", 1)
                .html(`
                    <strong>Household ${d.household_id}</strong><br>
                    Type: ${d.household_type}<br>
                    Wealth: ${d.wealth_level}<br>
                    Consumption: ${d3.format(",.2f")(d.consumption_kwh)} kWh<br>
                    Generation: ${d3.format(",.2f")(d.generation_kwh)} kWh<br>
                    Self-consumption: ${formatPercent(d.self_consumption_ratio)}<br>
                    Net cost: ${formatCurrency(d.net_cost)}
                `)
                .style("left", `${event.pageX + 14}px`)
                .style("top", `${event.pageY - 32}px`);
        })
        .on("mouseleave", () => {
            circles
                .attr("opacity", d => {
                    const householdMatch = filters.householdType === "all" || d.household_type === filters.householdType;
                    const wealthMatch = filters.wealthLevel === "all" || d.wealth_level === filters.wealthLevel;
                    return householdMatch && wealthMatch ? 0.9 : 0.12;
                })
                .attr("stroke-width", 1.2);

            tooltip.style("opacity", 0);
        });

    const legendData = [
        { label: "low", color: "#60a5fa" },
        { label: "middle", color: "#34d399" },
        { label: "high", color: "#f59e0b" },
        { label: "luxury", color: "#f472b6" }
    ];

    const legend = g.selectAll(".scatter-legend")
        .data(legendData)
        .enter()
        .append("g")
        .attr("transform", (d, i) => `translate(${innerWidth - 10}, ${200 + i * 24})`);

    legend.append("rect")
        .attr("width", 14)
        .attr("height", 14)
        .attr("fill", d => d.color);

    legend.append("text")
        .attr("x", 22)
        .attr("y", 12)
        .attr("class", "legend")
        .text(d => d.label);
}

function drawTimeseriesByType(containerSelector, data, timescale) {
    if (!data || data.length === 0) {
        showErrorInContainer(containerSelector, "No data available for Timeseries by Household Type.");
        return;
    }

    const aggregatedByType = [];

    const grouped = d3.group(data, d => d.household_type);

    grouped.forEach((rows, type) => {
        const agg = aggregateTimeSeries(rows, timescale, ["load_kwh"]);
        agg.forEach(row => {
            aggregatedByType.push({
                time: row.time,
                household_type: type,
                load_kwh: row.load_kwh
            });
        });
    });

    const { g, innerWidth, innerHeight } = createSvg(
        containerSelector,
        1200,
        450,
        { top: 30, right: 60, bottom: 70, left: 80 }
    );

    const householdTypes = Array.from(new Set(aggregatedByType.map(d => d.household_type)));

    const x = d3.scaleLinear()
        .domain(d3.extent(aggregatedByType, d => d.time))
        .range([0, innerWidth]);

    const y = d3.scaleLinear()
        .domain([0, d3.max(aggregatedByType, d => d.load_kwh) * 1.1 || 1])
        .range([innerHeight, 0]);

    const color = d3.scaleOrdinal()
        .domain(householdTypes)
        .range(d3.schemeTableau10);

    g.append("g")
        .attr("class", "axis")
        .attr("transform", `translate(0,${innerHeight})`)
        .call(d3.axisBottom(x));

    g.append("g")
        .attr("class", "axis")
        .call(d3.axisLeft(y));

    g.append("text")
        .attr("x", innerWidth / 2)
        .attr("y", innerHeight + 50)
        .attr("text-anchor", "middle")
        .attr("fill", "#d1d5db")
        .attr("font-size", 14)
        .text(
            timescale === "hourly"
                ? "Time Bucket (Hours)"
                : timescale === "daily"
                ? "Time Bucket (Days)"
                : "Time Bucket (Weeks)"
        );

    g.append("text")
        .attr("transform", "rotate(-90)")
        .attr("x", -innerHeight / 2)
        .attr("y", -55)
        .attr("text-anchor", "middle")
        .attr("fill", "#d1d5db")
        .attr("font-size", 14)
        .text("Consumption (kWh)");

    const line = d3.line()
        .x(d => x(d.time))
        .y(d => y(d.load_kwh));

    const linePaths = {};

    householdTypes.forEach(type => {
        const subset = aggregatedByType.filter(d => d.household_type === type);

        linePaths[type] = g.append("path")
            .datum(subset)
            .attr("fill", "none")
            .attr("stroke", color(type))
            .attr("stroke-width", 2.6)
            .attr("opacity", 0.95)
            .attr("d", line)
            .on("mouseenter", function () {
                Object.values(linePaths).forEach(path => path.attr("opacity", 0.18));
                linePaths[type].attr("opacity", 1).attr("stroke-width", 4);
            })
            .on("mouseleave", function () {
                Object.values(linePaths).forEach(path => path.attr("opacity", 0.95));
                linePaths[type].attr("stroke-width", 2.6);
            });

        g.selectAll(`.point-${type.replace(/\s+/g, "-")}`)
            .data(subset)
            .enter()
            .append("circle")
            .attr("cx", d => x(d.time))
            .attr("cy", d => y(d.load_kwh))
            .attr("r", 3.2)
            .attr("fill", color(type))
            .on("mousemove", (event, d) => {
                const maxValue = d3.max(subset, row => row.load_kwh);
                const minValue = d3.min(subset, row => row.load_kwh);
                const avgValue = d3.mean(subset, row => row.load_kwh);

                tooltip
                    .style("opacity", 1)
                    .html(`
                        <strong>${type}</strong><br>
                        Bucket: ${d.time}<br>
                        Consumption: ${d3.format(",.2f")(d.load_kwh)} kWh<br>
                        Avg (${type}): ${d3.format(",.2f")(avgValue)} kWh<br>
                        Min (${type}): ${d3.format(",.2f")(minValue)} kWh<br>
                        Max (${type}): ${d3.format(",.2f")(maxValue)} kWh
                    `)
                    .style("left", `${event.pageX + 14}px`)
                    .style("top", `${event.pageY - 34}px`);

                Object.values(linePaths).forEach(path => path.attr("opacity", 0.18));
                linePaths[type].attr("opacity", 1).attr("stroke-width", 4);
            })
            .on("mouseleave", () => {
                tooltip.style("opacity", 0);
                Object.values(linePaths).forEach(path => path.attr("opacity", 0.95));
                linePaths[type].attr("stroke-width", 2.6);
            });
    });

    const legend = g.selectAll(".timeseries-legend")
        .data(householdTypes)
        .enter()
        .append("g")
        .attr("transform", (d, i) => `translate(${innerWidth - 160}, ${15 + i * 24})`);

    legend.append("rect")
        .attr("width", 14)
        .attr("height", 14)
        .attr("fill", d => color(d));

    legend.append("text")
        .attr("x", 22)
        .attr("y", 12)
        .attr("class", "legend")
        .text(d => d);

    g.append("text")
        .attr("x", innerWidth)
        .attr("y", -8)
        .attr("text-anchor", "end")
        .attr("fill", "#94a3b8")
        .attr("font-size", 12)
        .text(`Grouped by: ${timescale}`);
}

function drawBatteryUtilization(containerSelector, data, filters) {
    const filtered = data
        .filter(d => filters.householdType === "all" || d.household_type === filters.householdType)
        .sort((a, b) => (+b.battery_discharge_kwh) - (+a.battery_discharge_kwh));

    const { g, innerWidth, innerHeight } = createSvg(containerSelector, 700, 420);

    const x = d3.scaleBand()
        .domain(filtered.map(d => d.household_type))
        .range([0, innerWidth])
        .padding(0.25);

    const y = d3.scaleLinear()
        .domain([0, d3.max(filtered, d => +d.battery_discharge_kwh) * 1.1 || 1])
        .range([innerHeight, 0]);

    g.append("g")
        .attr("class", "axis")
        .attr("transform", `translate(0,${innerHeight})`)
        .call(d3.axisBottom(x));

    g.append("g")
        .attr("class", "axis")
        .call(d3.axisLeft(y));

    g.selectAll("rect")
        .data(filtered)
        .enter()
        .append("rect")
        .attr("x", d => x(d.household_type))
        .attr("y", d => y(+d.battery_discharge_kwh))
        .attr("width", x.bandwidth())
        .attr("height", d => innerHeight - y(+d.battery_discharge_kwh))
        .attr("fill", "#a78bfa")
        .on("mousemove", (event, d) => {
            tooltip
                .style("opacity", 1)
                .html(`
                    <strong>${d.household_type}</strong><br>
                    Battery discharge: ${formatKwh(+d.battery_discharge_kwh)}
                `)
                .style("left", `${event.pageX + 12}px`)
                .style("top", `${event.pageY - 28}px`);
        })
        .on("mouseleave", () => tooltip.style("opacity", 0));
}

function drawSurplusDeficit(containerSelector, data, timescaleLabel) {
    const formatted = data.map(d => ({
        time: +d.time,
        net_load_kwh: +d.net_load_kwh
    }));

    const { g, innerWidth, innerHeight } = createSvg(containerSelector, 700, 420);

    const x = d3.scaleBand()
        .domain(formatted.map(d => d.time))
        .range([0, innerWidth])
        .padding(0.1);

    const maxAbs = d3.max(formatted, d => Math.abs(d.net_load_kwh)) || 1;

    const y = d3.scaleLinear()
        .domain([-maxAbs * 1.1, maxAbs * 1.1])
        .range([innerHeight, 0]);

    g.append("g")
        .attr("class", "axis")
        .attr("transform", `translate(0,${y(0)})`)
        .call(d3.axisBottom(x).tickValues(x.domain().filter((d, i) => i % Math.ceil(formatted.length / 10) === 0)));

    g.append("g")
        .attr("class", "axis")
        .call(d3.axisLeft(y));

    g.selectAll("rect")
        .data(formatted)
        .enter()
        .append("rect")
        .attr("x", d => x(d.time))
        .attr("width", x.bandwidth())
        .attr("y", d => d.net_load_kwh >= 0 ? y(d.net_load_kwh) : y(0))
        .attr("height", d => Math.abs(y(d.net_load_kwh) - y(0)))
        .attr("fill", d => d.net_load_kwh >= 0 ? "#ef4444" : "#22c55e")
        .on("mousemove", (event, d) => {
            tooltip
                .style("opacity", 1)
                .html(`
                    <strong>Bucket ${d.time}</strong><br>
                    Net load: ${formatKwh(d.net_load_kwh)}
                `)
                .style("left", `${event.pageX + 12}px`)
                .style("top", `${event.pageY - 28}px`);
        })
        .on("mouseleave", () => tooltip.style("opacity", 0));

    g.append("text")
        .attr("x", innerWidth)
        .attr("y", -8)
        .attr("text-anchor", "end")
        .attr("fill", "#94a3b8")
        .attr("font-size", 12)
        .text(`Grouped by: ${timescaleLabel}`);
}


function updateMLMetrics(metrics) {
    if (!metrics) return;

    document.getElementById("ml-r2").textContent =
        d3.format(".3f")(metrics.test_metrics.r2);

    document.getElementById("ml-rmse").textContent =
        d3.format(".3f")(metrics.test_metrics.rmse);

    document.getElementById("ml-mae").textContent =
        d3.format(".3f")(metrics.test_metrics.mae);

    document.getElementById("ml-train-rows").textContent =
        d3.format(",")(metrics.train_rows);

    document.getElementById("ml-test-rows").textContent =
        d3.format(",")(metrics.test_rows);
}


function buildMLSummaryText(data) {
    if (!data || data.length === 0) {
        return "No ML prediction data available.";
    }

    const avgActual = d3.mean(data, d => +d.actual_power_kw);
    const avgPredicted = d3.mean(data, d => +d.predicted_power_kw);

    const avgError = d3.mean(data, d =>
        Math.abs(+d.actual_power_kw - +d.predicted_power_kw)
    );

    const highestActual = d3.max(data, d => +d.actual_power_kw);
    const highestPredicted = d3.max(data, d => +d.predicted_power_kw);

    return `
        The machine learning model predicts an average solar generation of
        <span class="ml-summary-highlight">${d3.format(",.2f")(avgPredicted)} kW</span>,
        compared to a real average generation of
        <span class="ml-summary-highlight">${d3.format(",.2f")(avgActual)} kW</span>.
        The average prediction error is
        <span class="ml-summary-highlight">${d3.format(",.2f")(avgError)} kW</span>.
        Peak real generation reached
        <span class="ml-summary-highlight">${d3.format(",.2f")(highestActual)} kW</span>,
        while the model predicted peaks up to
        <span class="ml-summary-highlight">${d3.format(",.2f")(highestPredicted)} kW</span>.
    `;
}


function drawMLPredictionChart(containerSelector, data) {

    if (!data || data.length === 0) {
        showErrorInContainer(containerSelector, "No ML prediction data available.");
        return;
    }

    data.forEach((d, i) => {
        d.index = i;
        d.actual_power_kw = +d.actual_power_kw;
        d.predicted_power_kw = +d.predicted_power_kw;
    });

    const { g, innerWidth, innerHeight } = createSvg(
        containerSelector,
        1200,
        480,
        { top: 40, right: 40, bottom: 70, left: 80 }
    );

    const x = d3.scaleLinear()
        .domain([0, data.length - 1])
        .range([0, innerWidth]);

    const y = d3.scaleLinear()
        .domain([
            0,
            d3.max(data, d =>
                Math.max(d.actual_power_kw, d.predicted_power_kw)
            ) * 1.1
        ])
        .range([innerHeight, 0]);

    g.append("g")
        .attr("class", "axis")
        .attr("transform", `translate(0,${innerHeight})`)
        .call(d3.axisBottom(x));

    g.append("g")
        .attr("class", "axis")
        .call(d3.axisLeft(y));

    const actualLine = d3.line()
        .x(d => x(d.index))
        .y(d => y(d.actual_power_kw));

    const predictedLine = d3.line()
        .x(d => x(d.index))
        .y(d => y(d.predicted_power_kw));

    g.append("path")
        .datum(data)
        .attr("class", "ml-line-real")
        .attr("d", actualLine);

    g.append("path")
        .datum(data)
        .attr("class", "ml-line-predicted")
        .attr("d", predictedLine);

    const overlay = g.append("rect")
        .attr("width", innerWidth)
        .attr("height", innerHeight)
        .attr("fill", "transparent");

    const focusLine = g.append("line")
        .attr("stroke", "#94a3b8")
        .attr("stroke-width", 1)
        .attr("stroke-dasharray", "4,4")
        .attr("y1", 0)
        .attr("y2", innerHeight)
        .style("opacity", 0);

    overlay
        .on("mousemove", function(event) {

            const [mx] = d3.pointer(event, this);

            const hoveredIndex = Math.round(x.invert(mx));

            const d = data[
                Math.max(0, Math.min(data.length - 1, hoveredIndex))
            ];

            focusLine
                .style("opacity", 1)
                .attr("x1", x(d.index))
                .attr("x2", x(d.index));

            tooltip
                .style("opacity", 1)
                .html(`
                    <strong>Sample ${d.index}</strong><br>
                    Actual: ${d3.format(",.2f")(d.actual_power_kw)} kW<br>
                    Predicted: ${d3.format(",.2f")(d.predicted_power_kw)} kW<br>
                    Error: ${d3.format(",.2f")(
                        Math.abs(d.actual_power_kw - d.predicted_power_kw)
                    )} kW
                `)
                .style("left", `${event.pageX + 14}px`)
                .style("top", `${event.pageY - 32}px`);
        })
        .on("mouseleave", function() {
            tooltip.style("opacity", 0);
            focusLine.style("opacity", 0);
        });

    const legendData = [
        { label: "Actual Generation", color: "#38bdf8" },
        { label: "Predicted Generation", color: "#22c55e" }
    ];

    const legend = g.selectAll(".ml-legend-group")
        .data(legendData)
        .enter()
        .append("g")
        .attr("transform", (d, i) =>
            `translate(${innerWidth - 220}, ${15 + i * 24})`
        );

    legend.append("rect")
        .attr("width", 14)
        .attr("height", 14)
        .attr("fill", d => d.color);

    legend.append("text")
        .attr("x", 22)
        .attr("y", 12)
        .attr("class", "ml-legend")
        .text(d => d.label);

    document.getElementById("ml-summary-text").innerHTML =
        buildMLSummaryText(data);
}

function drawMLResidualChart(containerSelector, data) {
    if (!data || data.length === 0) {
        showErrorInContainer(containerSelector, "No residual error data available.");
        return;
    }

    data.forEach((d, i) => {
        d.index = i;
        d.error_kw = +d.error_kw;
        d.absolute_error_kw = +d.absolute_error_kw;
    });

    const { g, innerWidth, innerHeight } = createSvg(
        containerSelector,
        1200,
        460,
        { top: 40, right: 40, bottom: 70, left: 80 }
    );

    const x = d3.scaleLinear()
        .domain([0, data.length - 1])
        .range([0, innerWidth]);

    const maxAbsError = d3.max(data, d => Math.abs(d.error_kw)) || 1;

    const y = d3.scaleLinear()
        .domain([-maxAbsError * 1.1, maxAbsError * 1.1])
        .range([innerHeight, 0]);

    g.append("g")
        .attr("class", "axis")
        .attr("transform", `translate(0,${y(0)})`)
        .call(d3.axisBottom(x));

    g.append("g")
        .attr("class", "axis")
        .call(d3.axisLeft(y));

    g.append("line")
        .attr("x1", 0)
        .attr("x2", innerWidth)
        .attr("y1", y(0))
        .attr("y2", y(0))
        .attr("stroke", "#94a3b8")
        .attr("stroke-width", 1.2)
        .attr("stroke-dasharray", "5,5");

    g.selectAll("rect")
        .data(data)
        .enter()
        .append("rect")
        .attr("x", d => x(d.index))
        .attr("width", Math.max(1, innerWidth / data.length))
        .attr("y", d => d.error_kw >= 0 ? y(d.error_kw) : y(0))
        .attr("height", d => Math.abs(y(d.error_kw) - y(0)))
        .attr("fill", d => d.error_kw >= 0 ? "#38bdf8" : "#f97316")
        .attr("opacity", 0.85)
        .on("mousemove", (event, d) => {
            tooltip
                .style("opacity", 1)
                .html(`
                    <strong>Sample ${d.index}</strong><br>
                    Error: ${d3.format(",.2f")(d.error_kw)} kW<br>
                    Absolute error: ${d3.format(",.2f")(d.absolute_error_kw)} kW
                `)
                .style("left", `${event.pageX + 14}px`)
                .style("top", `${event.pageY - 32}px`);
        })
        .on("mouseleave", () => tooltip.style("opacity", 0));

    const avgAbsError = d3.mean(data, d => d.absolute_error_kw);
    const maxError = d3.max(data, d => d.error_kw);
    const minError = d3.min(data, d => d.error_kw);

    document.getElementById("ml-residual-summary-text").innerHTML =
        `The model has an average absolute error of <span class="ml-summary-highlight">${d3.format(",.2f")(avgAbsError)} kW</span>. `
        + `The largest underprediction is <span class="ml-summary-highlight">${d3.format(",.2f")(maxError)} kW</span>, `
        + `and the largest overprediction is <span class="ml-summary-highlight">${d3.format(",.2f")(Math.abs(minError))} kW</span>.`;
}


function drawIdealVsMLChart(containerSelector, data) {
    if (!data || data.length === 0) {
        showErrorInContainer(containerSelector, "No ideal vs ML data available.");
        return;
    }

    data.forEach((d, i) => {
        d.index = i;
        d.ideal_power_kw = +d.ideal_power_kw;
        d.predicted_power_kw = +d.predicted_power_kw;
    });

    const { g, innerWidth, innerHeight } = createSvg(
        containerSelector,
        1200,
        480,
        { top: 40, right: 40, bottom: 70, left: 80 }
    );

    const x = d3.scaleLinear()
        .domain([0, data.length - 1])
        .range([0, innerWidth]);

    const y = d3.scaleLinear()
        .domain([
            0,
            d3.max(data, d => Math.max(d.ideal_power_kw, d.predicted_power_kw)) * 1.1 || 1
        ])
        .range([innerHeight, 0]);

    g.append("g")
        .attr("class", "axis")
        .attr("transform", `translate(0,${innerHeight})`)
        .call(d3.axisBottom(x));

    g.append("g")
        .attr("class", "axis")
        .call(d3.axisLeft(y));

    const idealLine = d3.line()
        .x(d => x(d.index))
        .y(d => y(d.ideal_power_kw));

    const mlLine = d3.line()
        .x(d => x(d.index))
        .y(d => y(d.predicted_power_kw));

    g.append("path")
        .datum(data)
        .attr("fill", "none")
        .attr("stroke", "#f59e0b")
        .attr("stroke-width", 2.7)
        .attr("d", idealLine);

    g.append("path")
        .datum(data)
        .attr("fill", "none")
        .attr("stroke", "#22c55e")
        .attr("stroke-width", 2.7)
        .attr("stroke-dasharray", "8,4")
        .attr("d", mlLine);

    const legendData = [
        { label: "Ideal Solar Curve", color: "#f59e0b" },
        { label: "ML Solar Prediction", color: "#22c55e" }
    ];

    const legend = g.selectAll(".ideal-ml-legend")
        .data(legendData)
        .enter()
        .append("g")
        .attr("transform", (d, i) => `translate(${innerWidth - 220}, ${15 + i * 24})`);

    legend.append("rect")
        .attr("width", 14)
        .attr("height", 14)
        .attr("fill", d => d.color);

    legend.append("text")
        .attr("x", 22)
        .attr("y", 12)
        .attr("class", "legend")
        .text(d => d.label);

    const avgIdeal = d3.mean(data, d => d.ideal_power_kw);
    const avgML = d3.mean(data, d => d.predicted_power_kw);

    document.getElementById("ideal-vs-ml-summary-text").innerHTML =
        `The ideal curve assumes an average generation of <span class="ml-summary-highlight">${d3.format(",.2f")(avgIdeal)} kW</span>, `
        + `while the ML model predicts <span class="ml-summary-highlight">${d3.format(",.2f")(avgML)} kW</span> on average. `
        + `This shows how weather-based prediction makes the simulation less perfect but more realistic.`;
}


function drawForecastComparisonChart(containerSelector, data) {
    if (!data || data.length === 0) {
        showErrorInContainer(containerSelector, "No forecast comparison data available.");
        return;
    }

    data.forEach((d, i) => {
        d.index = i;
        d.actual_power_kw = +d.actual_power_kw;
        d.ml_power_kw = +d.ml_power_kw;
        d.day_ahead_power_kw = +d.day_ahead_power_kw;
        d.four_hours_ahead_power_kw = +d.four_hours_ahead_power_kw;
    });

    const { g, innerWidth, innerHeight } = createSvg(
        containerSelector,
        1200,
        500,
        { top: 40, right: 40, bottom: 70, left: 80 }
    );

    const x = d3.scaleLinear()
        .domain([0, data.length - 1])
        .range([0, innerWidth]);

    const y = d3.scaleLinear()
        .domain([
            0,
            d3.max(data, d => Math.max(
                d.actual_power_kw,
                d.ml_power_kw,
                d.day_ahead_power_kw,
                d.four_hours_ahead_power_kw
            )) * 1.1 || 1
        ])
        .range([innerHeight, 0]);

    g.append("g")
        .attr("class", "axis")
        .attr("transform", `translate(0,${innerHeight})`)
        .call(d3.axisBottom(x));

    g.append("g")
        .attr("class", "axis")
        .call(d3.axisLeft(y));

    const series = [
        { key: "actual_power_kw", label: "Actual", color: "#38bdf8", dash: null },
        { key: "ml_power_kw", label: "ML", color: "#22c55e", dash: "8,4" },
        { key: "day_ahead_power_kw", label: "Day Ahead", color: "#f59e0b", dash: "4,4" },
        { key: "four_hours_ahead_power_kw", label: "4 Hours Ahead", color: "#a78bfa", dash: "2,4" }
    ];

    const line = key => d3.line()
        .x(d => x(d.index))
        .y(d => y(d[key]));

    series.forEach(s => {
        g.append("path")
            .datum(data)
            .attr("fill", "none")
            .attr("stroke", s.color)
            .attr("stroke-width", 2.4)
            .attr("stroke-dasharray", s.dash)
            .attr("opacity", 0.9)
            .attr("d", line(s.key));
    });

    const legend = g.selectAll(".forecast-legend")
        .data(series)
        .enter()
        .append("g")
        .attr("transform", (d, i) => `translate(${innerWidth - 220}, ${15 + i * 24})`);

    legend.append("rect")
        .attr("width", 14)
        .attr("height", 14)
        .attr("fill", d => d.color);

    legend.append("text")
        .attr("x", 22)
        .attr("y", 12)
        .attr("class", "legend")
        .text(d => d.label);

    const maeML = d3.mean(data, d => Math.abs(d.actual_power_kw - d.ml_power_kw));
    const maeDA = d3.mean(data, d => Math.abs(d.actual_power_kw - d.day_ahead_power_kw));
    const maeHA4 = d3.mean(data, d => Math.abs(d.actual_power_kw - d.four_hours_ahead_power_kw));

    const best = [
        { name: "ML model", mae: maeML },
        { name: "Day Ahead forecast", mae: maeDA },
        { name: "4 Hours Ahead forecast", mae: maeHA4 }
    ].sort((a, b) => a.mae - b.mae)[0];

    document.getElementById("forecast-summary-text").innerHTML =
        `Average absolute error: ML = <span class="ml-summary-highlight">${d3.format(",.2f")(maeML)} kW</span>, `
        + `Day Ahead = <span class="ml-summary-highlight">${d3.format(",.2f")(maeDA)} kW</span>, `
        + `4 Hours Ahead = <span class="ml-summary-highlight">${d3.format(",.2f")(maeHA4)} kW</span>. `
        + `The best performer in this sample is <span class="ml-summary-highlight">${best.name}</span>.`;
}

function setupMLTabs() {
    const tabButtons = document.querySelectorAll(".ml-tab-btn");
    const tabPanels = document.querySelectorAll(".ml-tab-panel");

    tabButtons.forEach(button => {
        button.addEventListener("click", () => {
            const selectedTab = button.dataset.tab;

            tabButtons.forEach(btn => btn.classList.remove("active"));
            tabPanels.forEach(panel => panel.classList.remove("active"));

            button.classList.add("active");

            const activePanel = document.getElementById(`ml-tab-${selectedTab}`);
            if (activePanel) {
                activePanel.classList.add("active");
            }

            renderAllCharts(dashboardState.rawDatasets);
        });
    });
}

function setupAnalysisTabs() {
    const tabButtons = document.querySelectorAll(".analysis-tab-btn");
    const tabPanels = document.querySelectorAll(".analysis-tab-panel");

    tabButtons.forEach(button => {
        button.addEventListener("click", () => {
            const selectedTab = button.dataset.tab;

            tabButtons.forEach(btn => btn.classList.remove("active"));
            tabPanels.forEach(panel => panel.classList.remove("active"));

            button.classList.add("active");

            const activePanel = document.getElementById(`analysis-tab-${selectedTab}`);
            if (activePanel) {
                activePanel.classList.add("active");
            }

            renderAllCharts(dashboardState.rawDatasets);
        });
    });
}

function updateExecutiveSummary(kpis, allResultsSummary, mlMetrics, forecastComparison) {
    if (!kpis || !allResultsSummary || !mlMetrics || !forecastComparison) return;

    const currentKpis = kpis[0];

    const currentSeason = seasonSelect.value;

    const sameSeason = allResultsSummary.filter(row => row.season === currentSeason);

    const bestStrategy = sameSeason.reduce((best, row) =>
        row.total_net_cost < best.total_net_cost ? row : best
    );

    document.getElementById("exec-best-strategy").textContent = bestStrategy.strategy;

    document.getElementById("exec-best-strategy-text").textContent =
        `${bestStrategy.strategy} has the lowest total net cost for ${currentSeason}, `
        + `with ${formatCurrency(bestStrategy.total_net_cost)} total net cost.`;

    const maeML = d3.mean(forecastComparison, d =>
        Math.abs(+d.actual_power_kw - +d.ml_power_kw)
    );

    const maeDA = d3.mean(forecastComparison, d =>
        Math.abs(+d.actual_power_kw - +d.day_ahead_power_kw)
    );

    const maeHA4 = d3.mean(forecastComparison, d =>
        Math.abs(+d.actual_power_kw - +d.four_hours_ahead_power_kw)
    );

    const bestForecast = [
        { name: "ML Model", mae: maeML },
        { name: "Day Ahead", mae: maeDA },
        { name: "4 Hours Ahead", mae: maeHA4 }
    ].sort((a, b) => a.mae - b.mae)[0];

    document.getElementById("exec-best-forecast").textContent = bestForecast.name;

    document.getElementById("exec-best-forecast-text").textContent =
        `${bestForecast.name} shows the lowest average absolute error in the current sample: `
        + `${d3.format(",.2f")(bestForecast.mae)} kW.`;

    const r2 = mlMetrics.test_metrics.r2;
    const rmse = mlMetrics.test_metrics.rmse;
    const mae = mlMetrics.test_metrics.mae;

    document.getElementById("exec-model-accuracy").textContent =
        `R² ${d3.format(".3f")(r2)}`;

    document.getElementById("exec-model-accuracy-text").textContent =
        `The ML model explains part of the solar generation behavior using weather data. `
        + `Test RMSE is ${d3.format(".3f")(rmse)} MW and MAE is ${d3.format(".3f")(mae)} MW.`;

    const importRatio =
        currentKpis.total_grid_import_kwh / currentKpis.total_consumption_kwh;

    document.getElementById("exec-grid-dependence").textContent =
        formatPercent(importRatio);

    document.getElementById("exec-grid-dependence-text").textContent =
        `Under the selected strategy, the neighborhood imports about `
        + `${formatPercent(importRatio)} of its consumed energy from the grid.`;
}

strategySelect.addEventListener("change", loadDashboard);
seasonSelect.addEventListener("change", loadDashboard);
timescaleSelect.addEventListener("change", loadDashboard);
householdFilter.addEventListener("change", loadDashboard);
wealthFilter.addEventListener("change", loadDashboard);
sortSelect.addEventListener("change", loadDashboard);
sortOrderSelect.addEventListener("change", loadDashboard);

duckPlayBtn.addEventListener("click", playDuckAnimation);
duckResetBtn.addEventListener("click", resetDuckAnimation);
duckSlider.addEventListener("input", () => {
    stopDuckAnimation();
    dashboardState.duckAnimationIndex = +duckSlider.value;
    duckSliderLabel.textContent = formatDuckSliderLabel(
        duckSlider.value,
        duckSlider.max,
        getCurrentFilters().timescale
    );
    renderAllCharts(dashboardState.rawDatasets);
});

setupMLTabs();
setupAnalysisTabs();
loadDashboard();