const API_BASE = "http://127.0.0.1:5000";

let visitorChart = null;
let monthChart = null;
let scatterChart = null;

const $ = (id) => document.getElementById(id);

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.error || `Request failed: ${response.status}`);
  }

  return data;
}

function showMessage(message) {
  console.error(message);
  alert(message);
}

function formatPercent(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(2)}%` : "N/A";
}

function setText(id, value) {
  const element = $(id);
  if (element) element.textContent = value;
}

function getChartContext(id) {
  const canvas = $(id);
  return canvas ? canvas.getContext("2d") : null;
}

function makeBarChart(canvasId, labels, values, label) {
  const ctx = getChartContext(canvasId);
  if (!ctx) return null;

  return new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label,
        data: values,
        backgroundColor: "#78aef0",
        borderColor: "#478fe0",
        borderWidth: 1,
        borderRadius: 5,
        maxBarThickness: 55
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { display: true, position: "top" }
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          title: {
            display: true,
            text: "Purchase Rate (%)"
          },
          ticks: {
            callback: value => `${value}%`
          }
        },
        x: {
          ticks: {
            autoSkip: false,
            maxRotation: 35,
            minRotation: 0
          }
        }
      }
    }
  });
}

function makeScatterChart(points) {
  const ctx = getChartContext("scatterChart");
  if (!ctx) return null;

  return new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [{
        label: "Customer Sessions",
        data: points,
        backgroundColor: "#8172ff",
        borderColor: "#6755e8",
        pointRadius: 3,
        pointHoverRadius: 5
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { display: true }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: "Product-Related Pages"
          },
          beginAtZero: true
        },
        y: {
          title: {
            display: true,
            text: "Product-Related Duration"
          },
          beginAtZero: true
        }
      }
    }
  });
}

function renderChart(canvasId, oldChart, labels, values, label) {
  if (oldChart) oldChart.destroy();
  return makeBarChart(canvasId, labels, values, label);
}

async function loadAnalytics() {
  const refreshButton = $("refreshData");
  if (refreshButton) {
    refreshButton.disabled = true;
    refreshButton.textContent = "Loading...";
  }

  try {
    const data = await fetchJSON(`${API_BASE}/analytics`);

    setText("totalSessions", data.total_sessions ?? data.rows ?? "N/A");
    setText("purchaseRate", formatPercent(data.purchase_rate));
    setText("avgSession", data.average_session_duration ?? "N/A");
    setText("avgPages", data.average_pages_viewed ?? "N/A");

    const rates = data.category_purchase_rates || {};

    const visitorRows =
      rates.visitor_purchase_rate ||
      rates.visitor_type_purchase_rate ||
      [];

    const monthRows =
      rates.month_purchase_rate ||
      rates.monthly_purchase_rate ||
      [];

    if (visitorChart) visitorChart.destroy();
    visitorChart = makeBarChart(
      "deviceChart",
      visitorRows.map(item => item.category),
      visitorRows.map(item => Number(item.purchase_rate)),
      "Purchase Rate (%)"
    );

    if (monthChart) monthChart.destroy();
    monthChart = makeBarChart(
      "trafficChart",
      monthRows.map(item => item.category),
      monthRows.map(item => Number(item.purchase_rate)),
      "Purchase Rate (%)"
    );

    if (scatterChart) scatterChart.destroy();

    const scatterPoints = data.session_pages_data || data.scatter_data || [];
    scatterChart = makeScatterChart(
      scatterPoints
        .map(point => ({
          x: Number(point.x),
          y: Number(point.y)
        }))
        .filter(point => Number.isFinite(point.x) && Number.isFinite(point.y))
    );

    if (!visitorRows.length || !monthRows.length) {
      console.warn(
        "Some analytics charts have no data. Check the /analytics API response."
      );
    }
  } catch (error) {
    showMessage(
      `Analytics load nathi thayu. Backend server check kar.\n${error.message}`
    );
  } finally {
    if (refreshButton) {
      refreshButton.disabled = false;
      refreshButton.textContent = "Refresh Data";
    }
  }
}

function getPredictionPayload() {
  const fields = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend"
  ];

  const payload = {};

  for (const field of fields) {
    const element = $(field);
    if (!element) continue;

    if (field === "Month" || field === "VisitorType") {
      payload[field] = element.value;
    } else if (field === "Weekend") {
      payload[field] = element.value === "true";
    } else {
      const value = Number(element.value || 0);

      if (!Number.isFinite(value)) {
        throw new Error(`Invalid value for ${field}`);
      }

      payload[field] = value;
    }
  }

  return payload;
}

async function predictPurchase(event) {
  if (event) event.preventDefault();

  const button = $("predictButton");
  const resultValue = $("predictionValue");
  const resultText = $("predictionText");
  const meter = $("predictionMeter");

  try {
    const payload = getPredictionPayload();

    if (button) {
      button.disabled = true;
      button.textContent = "Predicting...";
    }

    const data = await fetchJSON(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const probability = Number(
      data.purchase_probability ??
      data.probability ??
      data.probability_percent
    );

    const percent = Number.isFinite(probability)
      ? (probability <= 1 ? probability * 100 : probability)
      : NaN;

    if (!Number.isFinite(percent)) {
      throw new Error("API response ma probability field nathi mali.");
    }

    if (resultValue) {
      resultValue.textContent = `${percent.toFixed(1)}%`;
    }

    if (resultText) {
      resultText.textContent =
        data.prediction_label ||
        data.prediction ||
        (percent >= 50
          ? "Model predicts a higher chance of purchase."
          : "Model predicts a lower chance of purchase.");
    }

    if (meter) {
      meter.style.width = `${Math.max(0, Math.min(100, percent))}%`;
    }
  } catch (error) {
    showMessage(`Prediction failed.\n${error.message}`);
  } finally {
    if (button) {
      button.disabled = false;
      button.textContent = "Predict Purchase";
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  if (typeof Chart === "undefined") {
    showMessage("Chart.js load nathi thayu. Internet connection check kar.");
    return;
  }

  const refreshButton = $("refreshData");
  if (refreshButton) {
    refreshButton.addEventListener("click", loadAnalytics);
  }

  const form = $("predictionForm");
  if (form) {
    form.addEventListener("submit", predictPurchase);
  }

  loadAnalytics();
});