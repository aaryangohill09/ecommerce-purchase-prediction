let deviceChart;
let trafficChart;
let scatterChart;


const $ = (id) => {

    return document.getElementById(id);

};



/* LOAD ANALYTICS */

async function loadAnalytics() {

    try {

        const response =
            await fetch(
                "http://127.0.0.1:5000/analytics"
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.error ||
                "Analytics error"
            );

        }



        /* KPI */

        $("totalSessions").textContent =
            data.summary.total_sessions
                .toLocaleString();


        $("purchaseRate").textContent =
            data.summary.purchase_rate
                .toFixed(1) + "%";


        $("avgDuration").textContent =
            data.summary.avg_session_duration
                .toFixed(1) + " min";


        $("avgPages").textContent =
            data.summary.avg_pages_viewed
                .toFixed(1);



        /* DESTROY OLD CHARTS */

        if (deviceChart) {

            deviceChart.destroy();

        }


        if (trafficChart) {

            trafficChart.destroy();

        }


        if (scatterChart) {

            scatterChart.destroy();

        }



        /* DEVICE CHART */

        deviceChart =
            new Chart(
                $("deviceChart"),
                {

                    type: "bar",

                    data: {

                        labels:
                            data.device.labels,

                        datasets: [

                            {

                                label:
                                    "Purchase rate %",

                                data:
                                    data.device.values

                            }

                        ]

                    },

                    options: {

                        responsive: true,

                        plugins: {

                            legend: {

                                display: false

                            }

                        },

                        scales: {

                            y: {

                                beginAtZero: true,

                                max: 100

                            }

                        }

                    }

                }

            );



        /* TRAFFIC CHART */

        trafficChart =
            new Chart(
                $("trafficChart"),
                {

                    type: "bar",

                    data: {

                        labels:
                            data.traffic.labels,

                        datasets: [

                            {

                                label:
                                    "Purchase rate %",

                                data:
                                    data.traffic.values

                            }

                        ]

                    },

                    options: {

                        responsive: true,

                        plugins: {

                            legend: {

                                display: false

                            }

                        },

                        scales: {

                            y: {

                                beginAtZero: true,

                                max: 100

                            }

                        }

                    }

                }

            );



        /* SCATTER */

        scatterChart =
            new Chart(
                $("scatterChart"),
                {

                    type: "scatter",

                    data: {

                        datasets: [

                            {

                                label:
                                    "Customer sessions",

                                data:
                                    data.scatter

                            }

                        ]

                    },

                    options: {

                        responsive: true,

                        scales: {

                            x: {

                                title: {

                                    display: true,

                                    text:
                                        "Pages Viewed"

                                }

                            },

                            y: {

                                title: {

                                    display: true,

                                    text:
                                        "Session Duration (min)"

                                }

                            }

                        }

                    }

                }

            );


    }

    catch (error) {

        console.error(error);


        $("formStatus").textContent =
            "Backend is not running. Start Flask and refresh.";

    }

}



/* PREDICTION */

$("predictionForm")
    .addEventListener(
        "submit",
        async (event) => {

            event.preventDefault();


            $("resultTitle").textContent =
                "Analyzing...";


            $("probability").textContent =
                "—";


            $("meterFill").style.width =
                "0%";


            $("formStatus").textContent =
                "";



            const payload = {

                device_type:
                    $("device_type").value,

                traffic_source:
                    $("traffic_source").value,

                pages_viewed:
                    Number(
                        $("pages_viewed").value
                    ),

                session_duration:
                    Number(
                        $("session_duration").value
                    ),

                previous_purchases:
                    Number(
                        $("previous_purchases").value
                    )

            };



            try {

                const response =
                    await fetch(
                        "http://127.0.0.1:5000/predict",
                        {

                            method: "POST",

                            headers: {

                                "Content-Type":
                                    "application/json"

                            },

                            body:
                                JSON.stringify(
                                    payload
                                )

                        }
                    );


                const data =
                    await response.json();


                if (!response.ok) {

                    throw new Error(
                        data.error ||
                        "Prediction failed"
                    );

                }



                const percentage =
                    data.purchase_probability *
                    100;



                $("probability").textContent =
                    percentage.toFixed(2) +
                    "%";


                $("meterFill").style.width =
                    percentage + "%";



                if (data.prediction === 1) {

                    $("resultTitle").textContent =
                        "Purchase Likely";


                    $("resultMessage").textContent =
                        "The model estimates a higher purchase probability for this session.";

                }

                else {

                    $("resultTitle").textContent =
                        "Purchase Unlikely";


                    $("resultMessage").textContent =
                        "The model estimates a lower purchase probability for this session.";

                }


            }

            catch (error) {

                console.error(error);


                $("resultTitle").textContent =
                    "Prediction unavailable";


                $("resultMessage").textContent =
                    error.message;

            }

        }

    );



/* REFRESH */

$("refreshBtn")
    .addEventListener(
        "click",
        loadAnalytics
    );



/* INITIAL LOAD */

loadAnalytics();