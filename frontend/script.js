const API = "http://127.0.0.1:5000";
let deviceChart, trafficChart, scatterChart;

async function loadAnalytics() {
  try {
    const response = await fetch(`${API}/analytics`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Analytics request failed");

    document.getElementById("totalSessions").textContent = data.kpis.total_sessions.toLocaleString();
    document.getElementById("purchaseRate").textContent = `${data.kpis.purchase_rate}%`;
    document.getElementById("avgSession").textContent = `${data.kpis.average_session_duration} min`;
    document.getElementById("avgPages").textContent = data.kpis.average_pages_viewed;

    if (deviceChart) deviceChart.destroy();
    if (trafficChart) trafficChart.destroy();
    if (scatterChart) scatterChart.destroy();

    deviceChart = new Chart(document.getElementById("deviceChart"), {
      type: "bar",
      data: {
        labels: data.device_purchase_rate.map(x => x.device_type),
        datasets: [{ label: "Purchase Rate %", data: data.device_purchase_rate.map(x => x.purchase_rate) }]
      },
      options: { responsive: true, scales: { y: { beginAtZero: true, max: 100 } } }
    });

    trafficChart = new Chart(document.getElementById("trafficChart"), {
      type: "bar",
      data: {
        labels: data.traffic_purchase_rate.map(x => x.traffic_source),
        datasets: [{ label: "Purchase Rate %", data: data.traffic_purchase_rate.map(x => x.purchase_rate) }]
      },
      options: { responsive: true, scales: { y: { beginAtZero: true, max: 100 } } }
    });

    scatterChart = new Chart(document.getElementById("scatterChart"), {
      type: "scatter",
      data: {
        datasets: [{
          label: "Sessions",
          data: data.session_pages_data.map(x => ({ x: x.pages_viewed, y: x.session_duration }))
        }]
      },
      options: {
        responsive: true,
        scales: {
          x: { title: { display: true, text: "Pages Viewed" } },
          y: { title: { display: true, text: "Session Duration" } }
        }
      }
    });
  } catch (error) {
    console.error(error);
    document.getElementById("totalSessions").textContent = "API";
  }
}

document.getElementById("predictionForm").addEventListener("submit", async (event) => {
  event.preventDefault();

  const payload = {
    device_type: document.getElementById("device_type").value,
    traffic_source: document.getElementById("traffic_source").value,
    pages_viewed: Number(document.getElementById("pages_viewed").value),
    session_duration: Number(document.getElementById("session_duration").value),
    previous_purchases: Number(document.getElementById("previous_purchases").value)
  };

  const result = document.getElementById("predictionLabel");
  result.textContent = "Calculating...";

  try {
    const response = await fetch(`${API}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Prediction failed");

    document.getElementById("probability").textContent = `${data.purchase_probability}%`;
    document.getElementById("meterFill").style.width = `${data.purchase_probability}%`;
    result.textContent = data.label;
  } catch (error) {
    result.textContent = error.message;
  }
});

document.getElementById("refreshBtn").addEventListener("click", loadAnalytics);
loadAnalytics();
