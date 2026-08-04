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
    localFolder: "assist2015",
    usedInPaper: true,
    sourceLabel: "ASSISTments 2015 Skill Builder",
    sourceUrl:
      "https://sites.google.com/site/assistmentsdata/home/2015-assistments-skill-builder-data",
    preprocessNotes:
      "Download the prepared processed package and place the extracted files under data/processed_datasets/assist2015/.",
    tag: "Paper dataset",
  },
  {
    name: "ASSIST2017",
    localFolder: "assist2017",
    usedInPaper: true,
    sourceLabel: "ASSISTments 2017 Dataset",
    sourceUrl: "https://sites.google.com/view/assistmentsdatamining/dataset",
    preprocessNotes:
      "Download the prepared processed package and place the extracted files under data/processed_datasets/assist2017/.",
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
    sourceUrl: "https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=507",
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
    sourceUrl: "https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=507",
    preprocessNotes:
      "Download the prepared package and place it under data/processed_datasets/statics/.",
    tag: "Additional supported dataset",
  },
];

const FIELD_MEANING = {
  used_in_paper: "Whether this dataset is used in the paper experiments.",
  dataset: "Dataset name and its role in this repository.",
  local_folder: "Folder name expected under data/processed_datasets/.",
  official_source: "Official dataset landing page or source project page.",
  setup_notes: "Where to place the downloaded files before running experiments.",
};

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

const tableBody = document.getElementById("table-body");
const paperOnlyToggle = document.getElementById("paper-only-toggle");
const datasetCount = document.getElementById("dataset-count");
const tooltip = document.getElementById("tooltip");

function renderRows(paperOnly = false) {
  const visible = sortedDatasets.filter((item) => !paperOnly || item.usedInPaper);

  tableBody.innerHTML = visible
    .map(
      (item) => `
        <tr class="${item.usedInPaper ? "paper-used" : ""}">
          <td>
            <span class="status-badge ${item.usedInPaper ? "yes" : "no"}">
              ${item.usedInPaper ? "Yes" : "No"}
            </span>
          </td>
          <td class="left">
            <span class="dataset-title">${item.name}</span>
            <span class="dataset-tag">${item.tag}</span>
          </td>
          <td><code class="folder-code">${item.localFolder}</code></td>
          <td class="left">
            <a href="${item.sourceUrl}" target="_blank" rel="noreferrer">${item.sourceLabel}</a>
          </td>
          <td class="left">
            ${item.preprocessNotes}
            <div class="note-pill">${item.usedInPaper ? "Used in paper" : "Additional dataset"}</div>
          </td>
        </tr>
      `
    )
    .join("");

  datasetCount.textContent = `${visible.length} dataset${visible.length > 1 ? "s" : ""}`;
}

function attachTooltips() {
  document.querySelectorAll(".info-dot").forEach((dot) => {
    const meaning = FIELD_MEANING[dot.dataset.field];
    if (!meaning) {
      return;
    }

    dot.addEventListener("mouseenter", () => {
      tooltip.textContent = meaning;
      tooltip.style.display = "block";
    });

    dot.addEventListener("mousemove", (event) => {
      tooltip.style.left = `${event.pageX + 12}px`;
      tooltip.style.top = `${event.pageY + 12}px`;
    });

    dot.addEventListener("mouseleave", () => {
      tooltip.style.display = "none";
    });
  });
}

paperOnlyToggle.addEventListener("change", (event) => {
  renderRows(event.target.checked);
});

renderRows(false);
attachTooltips();
