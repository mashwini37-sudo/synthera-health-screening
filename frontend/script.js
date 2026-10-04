const imageInput = document.getElementById("imageInput");
const screenButton = document.getElementById("screenButton");
const loading = document.getElementById("loading");
const result = document.getElementById("result");

const predictionText = document.getElementById("prediction");
const confidenceText = document.getElementById("confidence");

screenButton.addEventListener("click", async () => {

    const file = imageInput.files[0];

    if (!file) {
        alert("Please choose an image first.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    loading.classList.remove("hidden");
    result.classList.add("hidden");

    try {

        const response = await fetch(
            "http://127.0.0.1:8001/predict",
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Prediction failed");
        }

        predictionText.textContent = data.prediction;
        confidenceText.textContent = data.confidence + "%";

        result.classList.remove("hidden");

    } catch (error) {

        alert("Error connecting to Synthera backend: " + error.message);

    } finally {

        loading.classList.add("hidden");

    }
});
