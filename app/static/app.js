const resumeInput = document.querySelector("#resume");
const jobInput = document.querySelector("#job");
const fileInput = document.querySelector("#resume-file");
const analyzeButton = document.querySelector("#analyze");
const message = document.querySelector("#message");
const results = document.querySelector("#results");

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  message.textContent = "Reading PDF…";
  const formData = new FormData();
  formData.append("file", file);
  try {
    const response = await fetch("/api/extract-resume", { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Could not read PDF.");
    resumeInput.value = payload.text;
    message.textContent = "Resume imported.";
  } catch (error) {
    message.textContent = error.message;
  }
});

analyzeButton.addEventListener("click", async () => {
  message.textContent = "";
  results.hidden = true;
  analyzeButton.disabled = true;
  analyzeButton.firstChild.textContent = "Analysing evidence ";
  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ resume_text: resumeInput.value, job_description: jobInput.value }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail?.[0]?.msg || payload.detail || "Analysis failed.");
    renderResults(payload);
  } catch (error) {
    message.textContent = error.message;
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.firstChild.textContent = "Analyse evidence ";
  }
});

function renderResults(payload) {
  document.querySelector("#score").textContent = `${payload.summary.match_score}%`;
  document.querySelector("#supported-count").textContent = payload.summary.supported;
  document.querySelector("#partial-count").textContent = payload.summary.partial;
  document.querySelector("#missing-count").textContent = payload.summary.missing;
  document.querySelector("#disclaimer").textContent = payload.disclaimer;
  const list = document.querySelector("#match-list");
  list.replaceChildren(...payload.matches.map(createMatchCard));
  results.hidden = false;
  results.scrollIntoView({ behavior: "smooth", block: "start" });
}

function createMatchCard(match) {
  const card = document.createElement("article");
  card.className = "match-card";
  const top = document.createElement("div");
  top.className = "match-top";
  const title = document.createElement("h3");
  title.textContent = match.requirement;
  const status = document.createElement("span");
  status.className = `status ${match.status}`;
  status.textContent = match.status;
  top.append(title, status);
  card.append(top);

  if (match.evidence.length) {
    const evidence = document.createElement("div");
    evidence.className = "evidence";
    match.evidence.forEach((item) => {
      const quote = document.createElement("p");
      quote.textContent = `“${item.text}”`;
      evidence.append(quote);
    });
    card.append(evidence);
  }

  const recommendation = document.createElement("p");
  recommendation.className = "recommendation";
  recommendation.textContent = match.recommendation;
  card.append(recommendation);
  return card;
}

