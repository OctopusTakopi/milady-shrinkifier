from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from pipeline_common import CATALOG_PATH, LABELS, REVIEW_QUEUES, connect_db, load_review_items, queue_items


class LabelPayload(BaseModel):
    sha256: str
    label: str
    note: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local avatar review app for labeling Milady classifier data.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8765, help="Bind port.")
    return parser.parse_args()


app = FastAPI(title="Milady Shrinkifier Review")


@app.get("/")
def root() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.get("/api/summary")
def summary() -> JSONResponse:
    connection = connect_db()
    items = load_review_items(connection)
    counts = {queue_name: len(queue_items(items, queue_name)) for queue_name in REVIEW_QUEUES}
    label_counts: dict[str, int] = {label: 0 for label in LABELS}
    unlabeled = 0
    for item in items:
        if item.label is None:
            unlabeled += 1
        elif item.label in label_counts:
            label_counts[item.label] += 1
    return JSONResponse(
        {
            "catalogPath": str(CATALOG_PATH),
            "totalImages": len(items),
            "queueCounts": counts,
            "labelCounts": label_counts,
            "unlabeled": unlabeled,
            "canUndo": latest_label_event(connection) is not None,
        }
    )


@app.get("/api/queue")
def get_queue(
    queue: str = Query("unlabeled"),
    index: int = Query(0, ge=0),
) -> JSONResponse:
    connection = connect_db()
    items = queue_items(load_review_items(connection), queue)
    if not items:
        return JSONResponse({"queue": queue, "index": 0, "total": 0, "item": None})
    bounded_index = min(index, len(items) - 1)
    return JSONResponse(
        {
            "queue": queue,
            "index": bounded_index,
            "total": len(items),
            "item": items[bounded_index].to_dict(),
        }
    )


@app.get("/api/item/{sha256}")
def get_item(sha256: str) -> JSONResponse:
    connection = connect_db()
    item = review_item_by_sha(connection, sha256)
    if item is None:
        raise HTTPException(status_code=404, detail="Review item not found")
    return JSONResponse({"item": item.to_dict()})


@app.get("/api/history")
def get_history(limit: int = Query(24, ge=1, le=100)) -> JSONResponse:
    connection = connect_db()
    items_by_sha = {item.sha256: item for item in load_review_items(connection)}
    history = []
    for event in recent_label_events(connection, limit):
        item = items_by_sha.get(event["image_sha256"])
        history.append(
            {
                "eventId": int(event["id"]),
                "sha256": str(event["image_sha256"]),
                "createdAt": str(event["created_at"]),
                "newLabel": str(event["new_label"]),
                "previousLabel": event["previous_label"],
                "item": item.to_dict() if item else None,
            }
        )
    return JSONResponse({"history": history})


@app.post("/api/label")
def label_avatar(payload: LabelPayload) -> JSONResponse:
    if payload.label not in LABELS:
        raise HTTPException(status_code=400, detail=f"Unsupported label: {payload.label}")

    connection = connect_db()
    existing = connection.execute(
        """
        SELECT sha256, label, label_source, labeled_at, review_notes
        FROM images
        WHERE sha256 = ?
        """,
        (payload.sha256,),
    ).fetchone()
    if not existing:
        raise HTTPException(status_code=404, detail=f"Unknown avatar sha256: {payload.sha256}")

    connection.execute(
        """
        INSERT INTO label_events (
          image_sha256,
          previous_label,
          previous_label_source,
          previous_labeled_at,
          previous_review_notes,
          new_label,
          new_review_notes,
          created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (
            payload.sha256,
            existing["label"],
            existing["label_source"],
            existing["labeled_at"],
            existing["review_notes"],
            payload.label,
            payload.note,
        ),
    )
    connection.execute(
        """
        UPDATE images
        SET label = ?,
            label_source = 'manual',
            labeled_at = CURRENT_TIMESTAMP,
            review_notes = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE sha256 = ?
        """,
        (payload.label, payload.note, payload.sha256),
    )
    connection.commit()
    return JSONResponse({"ok": True})


@app.post("/api/undo")
def undo_last_label() -> JSONResponse:
    connection = connect_db()
    event = latest_label_event(connection)
    if event is None:
        raise HTTPException(status_code=409, detail="No label action to undo")

    connection.execute(
        """
        UPDATE images
        SET label = ?,
            label_source = ?,
            labeled_at = ?,
            review_notes = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE sha256 = ?
        """,
        (
            event["previous_label"],
            event["previous_label_source"],
            event["previous_labeled_at"],
            event["previous_review_notes"],
            event["image_sha256"],
        ),
    )
    connection.execute("DELETE FROM label_events WHERE id = ?", (event["id"],))
    connection.commit()
    item = review_item_by_sha(connection, str(event["image_sha256"]))
    return JSONResponse(
        {
            "ok": True,
            "undoneSha256": str(event["image_sha256"]),
            "item": item.to_dict() if item else None,
        }
    )


@app.get("/api/image/{sha256}")
def get_image(sha256: str) -> FileResponse:
    connection = connect_db()
    row = connection.execute("SELECT local_path FROM images WHERE sha256 = ?", (sha256,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Image not found")
    path = Path(str(row["local_path"]))
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image file missing on disk")
    return FileResponse(path)


INDEX_HTML = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Milady Review</title>
    <style>
      :root {
        color-scheme: light;
        font-family: ui-sans-serif, system-ui, sans-serif;
        background: #f7f7f5;
        color: #111;
      }
      body {
        margin: 0;
        padding: 20px;
      }
      .layout {
        display: grid;
        grid-template-columns: 220px minmax(320px, 440px) minmax(340px, 1fr);
        gap: 20px;
        align-items: start;
      }
      .panel {
        background: white;
        border: 1px solid #ddd;
        padding: 16px;
      }
      button,
      select {
        font: inherit;
      }
      button {
        padding: 8px 12px;
        border: 1px solid #111;
        background: white;
        cursor: pointer;
      }
      .actions {
        display: flex;
        gap: 8px;
        margin-top: 12px;
      }
      .actions button[disabled] {
        opacity: 0.45;
        cursor: default;
      }
      .history-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-top: 16px;
      }
      .history-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 8px;
        margin-top: 10px;
      }
      .history-thumb {
        display: block;
        padding: 0;
        border: 1px solid #d7d7d2;
        background: white;
      }
      .history-thumb img {
        width: 100%;
        aspect-ratio: 1;
        object-fit: cover;
      }
      .history-thumb span {
        display: block;
        padding: 4px 6px;
        font-size: 11px;
        text-align: left;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      .history-empty {
        color: #666;
        font-size: 12px;
        margin: 10px 0 0;
      }
      .status-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 0 0 12px;
      }
      .pill {
        display: inline-flex;
        align-items: center;
        padding: 5px 8px;
        border: 1px solid #d0d0cc;
        background: #f7f7f5;
        font-size: 12px;
        line-height: 1;
      }
      .pill[data-tone="warn"] {
        border-color: #d4692d;
        color: #9a4312;
        background: #fff2e8;
      }
      .pill[data-tone="bad"] {
        border-color: #d13a31;
        color: #a32620;
        background: #fff0ef;
      }
      .pill[data-tone="good"] {
        border-color: #1a7f37;
        color: #176c31;
        background: #effaf1;
      }
      img {
        width: 100%;
        max-width: 400px;
        display: block;
        background: #eee;
      }
      dl {
        display: grid;
        grid-template-columns: max-content 1fr;
        gap: 8px 12px;
        margin: 0;
      }
      dt {
        color: #666;
      }
      dd {
        margin: 0;
      }
      .hint {
        color: #666;
        font-size: 12px;
        margin-top: 8px;
      }
      a {
        color: #0f62fe;
      }
    </style>
  </head>
  <body>
    <div class="layout">
      <section class="panel">
        <h2>Queues</h2>
        <label>
          Queue
          <select id="queue"></select>
        </label>
        <p id="summary"></p>
        <div class="actions">
          <button id="undo">Undo last label</button>
        </div>
        <div class="hint">Hotkeys: 1=milady, 2=not_milady, 3=unclear, x=skip, z=undo</div>
      </section>
      <section class="panel">
        <h2 id="title">No item</h2>
        <div id="status-strip" class="status-strip"></div>
        <img id="preview" alt="avatar preview" />
        <div class="actions">
          <button data-label="milady">1 Milady</button>
          <button data-label="not_milady">2 Not Milady</button>
          <button data-label="unclear">3 Unclear</button>
          <button id="skip">Skip</button>
        </div>
        <div class="history-header">
          <h2>Recent labels</h2>
        </div>
        <div id="history-grid" class="history-grid"></div>
        <p id="history-empty" class="history-empty" hidden>No recent labels yet.</p>
      </section>
      <section class="panel">
        <h2>Metadata</h2>
        <dl id="metadata"></dl>
      </section>
    </div>
    <script>
      const queueSelect = document.getElementById("queue");
      const summaryNode = document.getElementById("summary");
      const preview = document.getElementById("preview");
      const title = document.getElementById("title");
      const statusStrip = document.getElementById("status-strip");
      const metadata = document.getElementById("metadata");
      const skip = document.getElementById("skip");
      const undo = document.getElementById("undo");
      const historyGrid = document.getElementById("history-grid");
      const historyEmpty = document.getElementById("history-empty");

      let index = 0;
      let selectedSha = null;

      async function loadSummary() {
        const response = await fetch("/api/summary");
        const payload = await response.json();
        summaryNode.textContent = `${payload.totalImages} images, ${payload.unlabeled} unlabeled`;
        queueSelect.innerHTML = Object.entries(payload.queueCounts)
          .map(([queue, count]) => `<option value="${queue}">${queue} (${count})</option>`)
          .join("");
        undo.disabled = !payload.canUndo;
      }

      async function loadItem() {
        if (selectedSha) {
          const response = await fetch(`/api/item/${encodeURIComponent(selectedSha)}`);
          if (response.ok) {
            const payload = await response.json();
            renderItem(payload.item, `selected ${selectedSha.slice(0, 8)}`);
            return;
          }
          selectedSha = null;
        }
        const queue = queueSelect.value || "unlabeled";
        const response = await fetch(`/api/queue?queue=${encodeURIComponent(queue)}&index=${index}`);
        const payload = await response.json();
        if (!payload.item) {
          title.textContent = "Queue empty";
          statusStrip.innerHTML = "";
          preview.removeAttribute("src");
          metadata.innerHTML = "";
          return;
        }
        index = payload.index;
        const item = payload.item;
        renderItem(item, `${payload.queue} ${payload.index + 1}/${payload.total}`);
      }

      function renderItem(item, heading) {
        title.textContent = heading;
        statusStrip.innerHTML = renderStatus(item);
        preview.src = `/api/image/${item.sha256}`;
        metadata.innerHTML = renderMetadata(item);
      }

      async function loadHistory() {
        const response = await fetch("/api/history");
        const payload = await response.json();
        historyGrid.innerHTML = "";
        historyEmpty.hidden = payload.history.length > 0;
        for (const entry of payload.history) {
          if (!entry.item) {
            continue;
          }
          const button = document.createElement("button");
          button.type = "button";
          button.className = "history-thumb";
          button.dataset.sha256 = entry.sha256;
          button.innerHTML = `<img src="/api/image/${entry.sha256}" alt="${entry.sha256}" /><span>${entry.newLabel}</span>`;
          button.addEventListener("click", async () => {
            selectedSha = entry.sha256;
            await loadItem();
          });
          historyGrid.append(button);
        }
      }

      function renderStatus(item) {
        const parts = [
          pill(`heuristic ${item.heuristicMatch ? "milady" : "not_milady"}`, item.heuristicMatch ? "warn" : "good"),
        ];
        if (item.latestModelPredictedLabel) {
          parts.push(
            pill(
              `model ${item.latestModelPredictedLabel}${item.maxModelScore != null ? ` ${formatScore(item.maxModelScore)}` : ""}`,
              item.latestModelPredictedLabel === "milady" ? "warn" : "good",
            ),
          );
        } else {
          parts.push(pill("model unscored"));
        }
        if (item.label) {
          parts.push(pill(`human ${item.label}`, item.label === "unclear" ? "warn" : item.label === "milady" ? "warn" : "good"));
        }
        for (const flag of item.disagreementFlags || []) {
          parts.push(pill(flag.replaceAll("_", " "), "bad"));
        }
        return parts.join("");
      }

      function pill(text, tone = "") {
        return `<span class="pill"${tone ? ` data-tone="${tone}"` : ""}>${text}</span>`;
      }

      function renderMetadata(item) {
        const rows = [
          ["sha256", item.sha256],
          ["label", item.label || "unlabeled"],
          ["handles", item.handles.join(", ") || "none"],
          ["display names", item.displayNames.join(", ") || "none"],
          ["seen count", item.seenCount],
          ["source surfaces", item.sourceSurfaces.join(", ") || "none"],
          ["heuristic", item.heuristicMatch ? `${item.heuristicSource || "match"} (${item.heuristicScore ?? "n/a"})` : "no"],
          ["model", item.latestModelPredictedLabel ? `${item.latestModelPredictedLabel} (${formatScore(item.maxModelScore)})` : "unscored"],
          ["model run", item.latestModelRunId || "n/a"],
          ["whitelisted", item.whitelisted ? "yes" : "no"],
          ["profile", item.exampleProfileUrl ? `<a href="${item.exampleProfileUrl}" target="_blank">${item.exampleProfileUrl}</a>` : "n/a"],
          ["tweet", item.exampleTweetUrl ? `<a href="${item.exampleTweetUrl}" target="_blank">${item.exampleTweetUrl}</a>` : "n/a"],
          ["notification", item.exampleNotificationUrl ? `<a href="${item.exampleNotificationUrl}" target="_blank">${item.exampleNotificationUrl}</a>` : "n/a"],
        ];
        return rows.map(([key, value]) => `<dt>${key}</dt><dd>${value}</dd>`).join("");
      }

      function formatScore(value) {
        if (value == null || Number.isNaN(Number(value))) {
          return "n/a";
        }
        return Number(value).toFixed(3);
      }

      async function labelCurrent(label) {
        const shaMatch = preview.src.match(/\\/api\\/image\\/([a-f0-9]+)/);
        if (!shaMatch) {
          return;
        }
        await fetch("/api/label", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ sha256: shaMatch[1], label }),
        });
        selectedSha = null;
        index += 1;
        await loadSummary();
        await loadHistory();
        await loadItem();
      }

      async function undoLast() {
        const response = await fetch("/api/undo", { method: "POST" });
        if (!response.ok) {
          return;
        }
        const payload = await response.json();
        selectedSha = payload.undoneSha256 || null;
        index = Math.max(0, index - 1);
        await loadSummary();
        await loadHistory();
        await loadItem();
      }

      queueSelect.addEventListener("change", async () => {
        selectedSha = null;
        index = 0;
        await loadItem();
      });
      skip.addEventListener("click", async () => {
        selectedSha = null;
        index += 1;
        await loadItem();
      });
      undo.addEventListener("click", undoLast);
      for (const button of document.querySelectorAll("button[data-label]")) {
        button.addEventListener("click", async () => {
          await labelCurrent(button.dataset.label);
        });
      }
      window.addEventListener("keydown", async (event) => {
        if (event.key === "1") await labelCurrent("milady");
        if (event.key === "2") await labelCurrent("not_milady");
        if (event.key === "3") await labelCurrent("unclear");
        if (event.key.toLowerCase() === "x") {
          selectedSha = null;
          index += 1;
          await loadItem();
        }
        if (event.key.toLowerCase() === "z") await undoLast();
      });
      loadSummary().then(async () => {
        await loadHistory();
        await loadItem();
      });
    </script>
  </body>
</html>
"""


def recent_label_events(connection, limit: int) -> list[Any]:
    return connection.execute(
        """
        SELECT id, image_sha256, previous_label, new_label, created_at
        FROM label_events
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def latest_label_event(connection):
    rows = recent_label_events(connection, 1)
    return rows[0] if rows else None


def review_item_by_sha(connection, sha256: str):
    for item in load_review_items(connection):
        if item.sha256 == sha256:
            return item
    return None


def main() -> None:
    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
