// SubPilot frontend — Phase 5.
// 原生 JS，无框架；所有动态内容用 textContent 渲染（防 XSS）。

"use strict";

const $ = (sel) => document.querySelector(sel);
const SESSION_ID = "ui-" + Math.random().toString(36).slice(2, 10);

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch (_) { /* 非 JSON 错误体 */ }
    throw new Error(detail);
  }
  return res.json();
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

/* ---------- 状态面板 ---------- */

const STATUS_TEXT = {
  before_school: "Before school",
  in_period: "In class",
  between_periods: "Between periods",
  after_school: "After school",
  no_classes: "No classes today",
};

function periodLabel(p) {
  return p.subject ? `${p.name} — ${p.subject}` : p.name;
}

function renderTopline(data) {
  const dateText = new Date(data.now).toLocaleDateString(undefined, {
    weekday: "long",
    month: "long",
    day: "numeric",
  });
  const s = data.schedule;
  let periodText = "";
  if (s && s.status === "in_period" && s.current_period) {
    periodText = ` · ${periodLabel(s.current_period)}`;
  }
  $("#topline").textContent = dateText + periodText;
}

function renderHero(data) {
  const s = data.schedule;
  const node = $("#hero-answer");
  if (!s) {
    node.textContent = "";
    return;
  }
  if (s.status === "in_period") {
    let text = `${periodLabel(s.current_period)} · ${s.minutes_remaining} min left`;
    if (s.next_period) text += ` · next: ${periodLabel(s.next_period)} at ${s.next_period.start}`;
    node.textContent = text;
  } else if (s.status === "between_periods") {
    node.textContent = `Break · next: ${periodLabel(s.next_period)} at ${s.next_period.start}`;
  } else if (s.status === "before_school") {
    node.textContent = `Before school · first: ${periodLabel(s.next_period)} at ${s.next_period.start}`;
  } else if (s.status === "after_school") {
    node.textContent = "After school — wrap up the day and write your note.";
  } else if (s.status === "no_classes") {
    node.textContent = "No classes today.";
  }
}

function renderNow(data) {
  const s = data.schedule;
  const body = $("#now-body");
  const timeline = $("#timeline");
  clear(body);
  clear(timeline);
  if (!s) {
    body.appendChild(el("p", "now-empty", "Quiet here until a schedule is loaded."));
    return;
  }

  const current = el("div", "now-current");
  const label = s.status === "in_period" && s.current_period
    ? periodLabel(s.current_period)
    : STATUS_TEXT[s.status] || s.status;
  current.textContent = label;
  body.appendChild(current);

  let sub = "";
  if (s.status === "in_period") {
    sub = `${s.current_period.start}–${s.current_period.end} · ${s.minutes_remaining} min remaining`;
    if (s.next_period) sub += ` · next ${periodLabel(s.next_period)} at ${s.next_period.start}`;
  } else if (s.next_period && s.minutes_until_next !== null) {
    sub = `next ${periodLabel(s.next_period)} at ${s.next_period.start} (in ${s.minutes_until_next} min)`;
  }
  if (sub) body.appendChild(el("p", "now-sub", sub));

  for (const p of s.periods_today) {
    const li = el("li");
    if (s.status === "in_period" && s.current_period && p.name === s.current_period.name) {
      li.classList.add("current");
    }
    li.appendChild(el("span", "t", `${p.start}–${p.end}`));
    li.appendChild(el("span", "name", periodLabel(p)));
    timeline.appendChild(li);
  }
}

function renderBathroom(data) {
  const list = $("#bathroom-list");
  clear(list);
  if (!data.bathroom.length) {
    list.appendChild(el("li", "empty", "No one is out."));
    return;
  }
  for (const p of data.bathroom) {
    const li = el("li");
    const name = el("span");
    name.appendChild(el("span", "dot"));
    name.appendChild(document.createTextNode(p.student));
    li.appendChild(name);
    li.appendChild(el("span", "meta", `out ${p.minutes_out} min · left ${p.left_at}`));
    list.appendChild(li);
  }
}

function renderEvents(data) {
  const list = $("#events-list");
  clear(list);
  if (!data.events.length) {
    list.appendChild(el("li", "empty", "Nothing logged yet."));
    return;
  }
  for (const e of data.events) {
    const li = el("li");
    li.appendChild(el("span", "t", e.time));
    if (e.period) li.appendChild(el("span", "period", `[${e.period}]`));
    li.appendChild(document.createTextNode(e.description));
    if (e.students && e.students.length) {
      li.appendChild(el("span", "students", ` — ${e.students.join(", ")}`));
    }
    list.appendChild(li);
  }
}

function renderDocuments(data) {
  const list = $("#doc-list");
  clear(list);
  if (!data.documents.length) {
    list.appendChild(el("li", "empty", "No documents yet."));
    return;
  }
  for (const d of data.documents) {
    const li = el("li");
    li.appendChild(el("span", "", d.name));
    li.appendChild(el("span", "meta", `${d.chunks} chunks`));
    list.appendChild(li);
  }
}

function renderNotes(data) {
  const box = $("#notes");
  clear(box);
  for (const note of data.notes || []) {
    box.appendChild(el("p", "", note));
  }
}

async function loadStatus() {
  try {
    const data = await api("/api/status");
    renderTopline(data);
    renderHero(data);
    renderNow(data);
    renderBathroom(data);
    renderEvents(data);
    renderDocuments(data);
    renderNotes(data);
  } catch (err) {
    renderNotes({ notes: ["Server unreachable: " + err.message] });
  }
}

/* ---------- 聊天 ---------- */

function addMessage(kind, text, sources) {
  const box = $("#messages");
  const msg = el("div", "message " + kind, text);
  if (sources && sources.length) {
    const foot = el("div", "sources");
    for (const s of sources) foot.appendChild(el("span", "", "Source: " + s));
    msg.appendChild(foot);
  }
  box.appendChild(msg);
  box.scrollTop = box.scrollHeight;
}

async function sendChat(text) {
  addMessage("user", text);
  const input = $("#chat-input");
  const btn = $("#send-btn");
  input.value = "";
  btn.disabled = true;
  const thinking = el("div", "message assistant thinking", "…");
  $("#messages").appendChild(thinking);
  $("#messages").scrollTop = $("#messages").scrollHeight;
  try {
    const body = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: SESSION_ID, text }),
    });
    addMessage("assistant", body.answer, body.sources);
    loadStatus(); // 工具可能记了 bathroom pass / event，立即刷新侧栏
  } catch (err) {
    addMessage("error", err.message);
  } finally {
    thinking.remove();
    btn.disabled = false;
    input.focus();
  }
}

/* ---------- 文档上传 ---------- */

const dropzone = $("#dropzone");
const fileInput = $("#file-input");

dropzone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  uploadFiles(fileInput.files);
  fileInput.value = "";
});

["dragenter", "dragover"].forEach((type) =>
  dropzone.addEventListener(type, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);

["dragleave", "drop"].forEach((type) =>
  dropzone.addEventListener(type, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);

dropzone.addEventListener("drop", (e) => uploadFiles(e.dataTransfer.files));

async function uploadFiles(files) {
  for (const file of files) {
    const form = new FormData();
    form.append("file", file);
    try {
      const result = await api("/api/documents", { method: "POST", body: form });
      addMessage("assistant", `Ingested ${result.name} (${result.chunks} chunks).`);
    } catch (err) {
      addMessage("error", `Could not upload ${file.name}: ${err.message}`);
    }
  }
  loadStatus();
}

/* ---------- 报告 ---------- */

async function generateReport() {
  const btn = $("#generate-report");
  btn.disabled = true;
  btn.textContent = "Generating…";
  try {
    const body = await api("/api/report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: "report" }),
    });
    $("#report-body").textContent = body.report;
    $("#report-letter").hidden = false;
  } catch (err) {
    addMessage("error", "Could not generate the note: " + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Generate note";
  }
}

/* ---------- 启动 ---------- */

$("#composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("#chat-input").value.trim();
  if (text) sendChat(text);
});

$("#generate-report").addEventListener("click", generateReport);

addMessage("assistant", "Good morning — I'm SubPilot. Ask me about the day, or tell me something to record.");

loadStatus();
setInterval(loadStatus, 60000);
