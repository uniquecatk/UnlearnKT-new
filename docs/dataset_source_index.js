const PROCESSED_PACKAGE_URL = "https://drive.google.com/drive/folders/14ZLY7B_Tgs8k82qW3eQD7ufcHh0Bq50W";

const DATASETS = [
  {
    name: "ASSIST2009",
    localFolder: "ASSIST2009",
    usedInPaper: true,
    sourceLabel: "ASSISTments 2009-2010 Skill Builder",
    sourceUrl:
      "https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010",
    preprocessNotes:
      "Download the official skill_builder_data.csv or your prepared single-file release, then place it directly under data/processed_datasets/ASSIST2009/.",
    tag: "Paper dataset",
  },
  {
    name: "ASSIST2015",
    localFolder: "assistments15",
    usedInPaper: true,
    sourceLabel: "ASSISTments 2015 Skill Builder",
    sourceUrl:
      "https://sites.google.com/site/assistmentsdata/home/2015-assistments-skill-builder-data",
    preprocessNotes:
      "Download the prepared processed package and place the extracted files under data/processed_datasets/assistments15/.",
    tag: "Paper dataset",
  },
  {
    name: "ASSIST2017",
    localFolder: "assistments17",
    usedInPaper: true,
    sourceLabel: "ASSISTments 2017 Dataset",
    sourceUrl:
      "https://sites.google.com/view/assistmentsdatamining/dataset",
    preprocessNotes:
      "Download the prepared processed package and place the extracted files under data/processed_datasets/assistments17/.",
    tag: "Paper dataset",
  },
  {
    name: "ASSIST2012",
    localFolder: "assist2012",
    usedInPaper: true,
    sourceLabel: "ASSISTments 2012-2013 School Data",
    sourceUrl:
      "https://sites.google.com/site/assistmentsdata/home/2012-13-school-data-with-affect",
    preprocessNotes:
      "Download the prepared package and place it under data/processed_datasets/assist2012/. Use the packaged benchmark-ready files directly.",
    tag: "Paper dataset",
  },
  {
    name: "STATICS2011",
    localFolder: "statics2011",
    usedInPaper: true,
    sourceLabel: "PSLC DataShop: Statics 2011",
    sourceUrl:
      "https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=507",
    preprocessNotes:
      "Download the prepared package and place it under data/processed_datasets/statics2011/. Use the packaged benchmark-ready files directly.",
    tag: "Paper dataset",
  },
  {
    name: "EdNet-KT1",
    localFolder: "ednet-kt1",
    usedInPaper: true,
    sourceLabel: "EdNet Official Repository",
    sourceUrl: "https://github.com/riiid/ednet",
    preprocessNotes:
      "Download the prepared package and place it under data/processed_datasets/ednet-kt1/. The repository uses the prepared benchmark-ready version directly.",
    tag: "Paper dataset",
  },
  {
    name: "ASSIST2012 (pyKT split variant)",
    localFolder: "assistments12",
    usedInPaper: false,
    sourceLabel: "ASSISTments 2012-2013 School Data",
    sourceUrl:
      "https://sites.google.com/site/assistmentsdata/home/2012-13-school-data-with-affect",
    preprocessNotes:
      "Download the prepared package and place it under data/processed_datasets/assistments12/.",
    tag: "Additional supported dataset",
  },
  {
    name: "Algebra 2005",
    localFolder: "algebra05",
    usedInPaper: false,
    sourceLabel: "PSLC DataShop: Algebra 2005-2006",
    sourceUrl: "https://pslcdatashop.web.cmu.edu/",
    preprocessNotes:
      "Download the prepared package and place it under data/processed_datasets/algebra05/.",
    tag: "Additional supported dataset",
  },
  {
    name: "Bridge to Algebra 2006",
    localFolder: "bridge_algebra06",
    usedInPaper: false,
    sourceLabel: "PSLC DataShop: Bridge to Algebra 2006-2007",
    sourceUrl: "https://pslcdatashop.web.cmu.edu/",
    preprocessNotes:
      "Download the prepared package and place it under data/processed_datasets/bridge_algebra06/.",
    tag: "Additional supported dataset",
  },
  {
    name: "Spanish",
    localFolder: "spanish",
    usedInPaper: false,
    sourceLabel: "Duolingo SLAM Shared Task",
    sourceUrl: "https://sharedtask.duolingo.com/",
    preprocessNotes:
      "Download the prepared package and place it under data/processed_datasets/spanish/.",
    tag: "Additional supported dataset",
  },
  {
    name: "STATICS (processed split variant)",
    localFolder: "statics",
    usedInPaper: false,
    sourceLabel: "PSLC DataShop: Statics 2011",
    sourceUrl:
      "https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=507",
    preprocessNotes:
      "Download the prepared package and place it under data/processed_datasets/statics/.",
    tag: "Additional supported dataset",
  },
];

const paperOrder = [
  "ASSIST2009",
  "ASSIST2015",
  "ASSIST2017",
  "ASSIST2012",
  "STATICS2011",
  "EdNet-KT1",
];

const sortedDatasets = [...DATASETS].sort((a, b) => {
  if (a.usedInPaper !== b.usedInPaper) {
    return a.usedInPaper ? -1 : 1;
  }

  if (a.usedInPaper && b.usedInPaper) {
    return paperOrder.indexOf(a.name) - paperOrder.indexOf(b.name);
  }

  return a.name.localeCompare(b.name);
});

const tableBody = document.getElementById("dataset-table-body");
const paperOnlyToggle = document.getElementById("paper-only-toggle");
const datasetCount = document.getElementById("dataset-count");

function renderPackageLink(url, label) {
  if (!url || url.startsWith("REPLACE_WITH_")) {
    return `<span class="muted">Replace placeholder with your Google Drive link</span>`;
  }

  return `<a href="${url}" target="_blank" rel="noreferrer">${label}</a>`;
}

function renderRows(paperOnly = false) {
  const visible = sortedDatasets.filter((item) => !paperOnly || item.usedInPaper);

  tableBody.innerHTML = visible
    .map(
      (item) => `
        <tr class="${item.usedInPaper ? "paper-used" : ""}">
          <td>
            <span class="check ${item.usedInPaper ? "yes" : "no"}">
              ${item.usedInPaper ? "✓" : "·"}
            </span>
          </td>
          <td>
            <span class="dataset-name">${item.name}</span>
            <span class="dataset-meta">${item.tag}</span>
          </td>
          <td><code class="folder">${item.localFolder}</code></td>
          <td>
            <a href="${item.sourceUrl}" target="_blank" rel="noreferrer">${item.sourceLabel}</a>
          </td>
          <td>${renderPackageLink(PROCESSED_PACKAGE_URL, "Download processed package")}</td>
          <td>
            ${item.preprocessNotes}
            <div class="pill">${item.usedInPaper ? "Used in the paper" : "Not used in the paper"}</div>
          </td>
        </tr>
      `
    )
    .join("");

  datasetCount.textContent = `${visible.length} dataset${visible.length > 1 ? "s" : ""} shown`;
}

paperOnlyToggle.addEventListener("change", (event) => {
  renderRows(event.target.checked);
});

renderRows(false);
