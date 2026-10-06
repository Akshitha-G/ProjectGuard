// Small shared helpers. Each page does its own fetch + render inline.

function badgeClass(category) {
  return {
    "Low": "low",
    "Moderate": "moderate",
    "High": "high",
    "Very High": "very-high",
  }[category] || "low";
}

function teamTag(project) {
  return `${project.section}${project.team_number}`;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

function fmtDate(iso) {
  try {
    return new Date(iso).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
  } catch {
    return iso;
  }
}
