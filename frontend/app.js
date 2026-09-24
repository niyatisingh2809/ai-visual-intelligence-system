const API_URL = "https://ai-visual-intelligence-system.onrender.com/api/analyze";

const imageInput = document.getElementById("imageInput");
const dropzone = document.getElementById("dropzone");
const previewWrap = document.getElementById("previewWrap");
const previewImage = document.getElementById("previewImage");
const changeButton = document.getElementById("changeButton");
const analyzeButton = document.getElementById("analyzeButton");
const statusBox = document.getElementById("status");
const results = document.getElementById("results");
const resultImage = document.getElementById("resultImage");
const resultImageEmpty = document.getElementById("resultImageEmpty");

let selectedFile = null;

function setStatus(message, isError = false) {
  statusBox.textContent = message;
  statusBox.classList.toggle("error", isError);
}

function setFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    setStatus("Please choose a valid image file.", true);
    return;
  }

  if (file.size > 10 * 1024 * 1024) {
    setStatus("Please choose an image smaller than 10 MB.", true);
    return;
  }

  selectedFile = file;
  previewImage.src = URL.createObjectURL(file);
  dropzone.classList.add("hidden");
  previewWrap.classList.remove("hidden");
  analyzeButton.disabled = false;
  results.classList.add("hidden");
  resultImage.classList.add("hidden");
  resultImageEmpty.classList.remove("hidden");
  setStatus("");
}

imageInput.addEventListener("change", () => {
  setFile(imageInput.files[0]);
});

changeButton.addEventListener("click", () => {
  imageInput.click();
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (event) => {
  setFile(event.dataTransfer.files[0]);
});

analyzeButton.addEventListener("click", async () => {
  if (!selectedFile) return;

  analyzeButton.disabled = true;
  analyzeButton.classList.add("loading");
  analyzeButton.textContent = "Analyzing…";
  setStatus("Connecting to the AI backend. The first request may take longer while Render wakes up.");

  const formData = new FormData();
  formData.append("file", selectedFile);

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 180000);

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });

    let data = null;
    try {
      data = await response.json();
    } catch (_) {}

    if (!response.ok) {
      throw new Error(
        (data && data.detail) || "Backend returned HTTP " + response.status + "."
      );
    }

    resultImage.src = data.boxed_image;
    resultImage.classList.remove("hidden");
    resultImageEmpty.classList.add("hidden");

    document.getElementById("captionResult").textContent = data.caption || "—";
    document.getElementById("objectsResult").textContent = data.objects_text || "—";
    document.getElementById("emotionResult").textContent = data.emotion || "—";
    document.getElementById("environmentResult").textContent = data.environment || "—";
    document.getElementById("descriptionResult").textContent = data.description || "—";
    document.getElementById("storyResult").textContent = data.story || "—";

    results.classList.remove("hidden");
    setStatus("Analysis complete.");
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    const message =
      error.name === "AbortError"
        ? "The backend took too long to respond. Please try again after a minute."
        : (error.message || "Could not reach the backend.");

    setStatus(message, true);
  } finally {
    clearTimeout(timeout);
    analyzeButton.disabled = false;
    analyzeButton.classList.remove("loading");
    analyzeButton.textContent = "Analyze image";
  }
});
