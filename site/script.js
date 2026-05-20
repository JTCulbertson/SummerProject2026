// ─────────────────────────────────────────────────────────────────────────────
// THEME TOGGLE
// ─────────────────────────────────────────────────────────────────────────────
(function () {
  const saved = localStorage.getItem("theme");
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const theme = saved || (prefersDark ? "dark" : "light");
  if (theme === "light") document.documentElement.dataset.theme = "light";

  function applyTheme(t) {
    if (t === "light") {
      document.documentElement.dataset.theme = "light";
    } else {
      delete document.documentElement.dataset.theme;
    }
    const moon = document.getElementById("icon-moon");
    const sun  = document.getElementById("icon-sun");
    if (moon && sun) {
      moon.style.display = t === "light" ? "none"  : "";
      sun.style.display  = t === "light" ? ""      : "none";
    }
    localStorage.setItem("theme", t);
  }

  applyTheme(theme);

  document.addEventListener("click", function (e) {
    if (e.target.closest("#theme-toggle")) {
      const current = document.documentElement.dataset.theme === "light" ? "light" : "dark";
      applyTheme(current === "light" ? "dark" : "light");
    }
  });
})();

// Updates are loaded from updates.json — edit that file to add new entries.

// ─────────────────────────────────────────────────────────────────────────────
// TAG STYLES — add new tag names here to give them a color class
// ─────────────────────────────────────────────────────────────────────────────
const TAG_CLASSES = {
  Research: "tag-research",
  Progress: "tag-progress",
  Setup:    "tag-setup",
  Results:  "tag-results",
  Bug:      "tag-bug",
};

// ─────────────────────────────────────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────────────────────────────────────
function tagClass(tag) {
  return TAG_CLASSES[tag] || "tag-default";
}

function formatDate(iso) {
  const d = new Date(iso + "T00:00:00");
  return d.toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });
}

function renderTags(tags) {
  return `<div class="tags">${tags.map(t =>
    `<span class="tag ${tagClass(t)}">${t}</span>`
  ).join("")}</div>`;
}

const BACK_ICON = `<svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
  <path fill-rule="evenodd" d="M7.78 12.53a.75.75 0 01-1.06 0L2.47 8.28a.75.75 0 010-1.06l4.25-4.25a.75.75 0 011.06 1.06L4.81 7h7.44a.75.75 0 010 1.5H4.81l2.97 2.97a.75.75 0 010 1.06z"/>
</svg>`;

// ─────────────────────────────────────────────────────────────────────────────
// RENDER: LIST VIEW
// ─────────────────────────────────────────────────────────────────────────────
function renderList(updates) {
  const cards = updates.map((u, i) => `
    <li>
      <a class="update-card" href="#${u.id}">
        <div class="card-meta">
          <span class="card-version">${u.version}</span>
          <span class="card-date">${formatDate(u.date)}</span>
          ${i === 0 ? '<span class="badge-latest">Latest</span>' : ""}
        </div>
        <h2 class="card-title">${u.title}</h2>
        <p class="card-excerpt">${u.excerpt}</p>
        ${renderTags(u.tags)}
      </a>
    </li>
  `).join("");

  return `
    <p class="list-eyebrow">Update Log</p>
    <h1 class="list-title">Summer Research 2026</h1>
    <ul class="update-list">${cards}</ul>
  `;
}

// ─────────────────────────────────────────────────────────────────────────────
// RENDER: SINGLE UPDATE VIEW
// ─────────────────────────────────────────────────────────────────────────────
function renderSections(sections) {
  return sections.map(s => {
    if (s.items) {
      return `<h3>${s.heading}</h3><ul>${s.items.map(i => `<li>${i}</li>`).join("")}</ul>`;
    }
    return `<h3>${s.heading}</h3><p>${s.body}</p>`;
  }).join("");
}

function renderUpdate(u) {
  return `
    <a class="back-link" href="#">${BACK_ICON} All Updates</a>
    <div class="update-header">
      <div class="card-meta">
        <span class="card-version">${u.version}</span>
        <span class="card-date">${formatDate(u.date)}</span>
        ${renderTags(u.tags)}
      </div>
      <h1 class="update-header-title">${u.title}</h1>
    </div>
    <div class="update-body">${renderSections(u.sections)}</div>
  `;
}

// ─────────────────────────────────────────────────────────────────────────────
// ROUTER — reads the URL hash to decide which view to show
// ─────────────────────────────────────────────────────────────────────────────
function route(updates) {
  const main = document.getElementById("main");
  const id   = location.hash.slice(1);

  if (id) {
    const update = updates.find(u => u.id === id);
    if (update) {
      main.innerHTML = renderUpdate(update);
      document.title = `${update.version}: ${update.title} — Summer Project 2026`;
      window.scrollTo(0, 0);
      return;
    }
  }

  main.innerHTML = renderList(updates);
  document.title = "Summer Project 2026 — Update Log";
}

// ─────────────────────────────────────────────────────────────────────────────
// BOOT — fetch updates.json then wire up routing
// ─────────────────────────────────────────────────────────────────────────────
fetch("updates.json")
  .then(r => r.json())
  .then(updates => {
    route(updates);
    window.addEventListener("hashchange", () => route(updates));
  })
  .catch(() => {
    document.getElementById("main").innerHTML =
      `<p style="color:var(--muted)">Could not load updates.json.</p>`;
  });
