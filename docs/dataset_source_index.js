const DATASETS = [
  {
    name: "ASSIST2009",
    usedInPaper: true,
    sourceLabel: "ASSISTments 2009-2010 Skill Builder",
    sourceUrl:
      "https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010",
    tag: "Paper dataset",
    nInteraction: "338001",
    nUser: "4163",
    nConcept: "123",
    nQuestion: "17751",
    avgSeqLen: "70.27",
    avgAnswerAcc: "0.6583",
    questionSparsity: "0.9961",
    conceptSparsity: "0.9156",
    sideInfo: "concept_text, school_id, timestamp, num_hint, num_attempt",
  },
  {
    name: "ASSIST2015",
    usedInPaper: true,
    sourceLabel: "ASSISTments 2015 Skill Builder",
    sourceUrl:
      "https://sites.google.com/site/assistmentsdata/home/2015-assistments-skill-builder-data",
    tag: "Paper dataset",
    nInteraction: "708631",
    nUser: "19917",
    nConcept: "100",
    nQuestion: "100",
    avgSeqLen: "36.46",
    avgAnswerAcc: "0.7295",
    questionSparsity: "0.9359",
    conceptSparsity: "0.9359",
    sideInfo: "-",
  },
  {
    name: "ASSIST2017",
    usedInPaper: true,
    sourceLabel: "ASSISTments 2017 Dataset",
    sourceUrl: "https://sites.google.com/view/assistmentsdatamining/dataset",
    tag: "Paper dataset",
    nInteraction: "864713",
    nUser: "1709",
    nConcept: "101",
    nQuestion: "2803",
    avgSeqLen: "505.98",
    avgAnswerAcc: "0.3674",
    questionSparsity: "0.9265",
    conceptSparsity: "0.5935",
    sideInfo: "concept_text, school_id, timestamp, use_time, num_hint, num_attempt",
  },
  {
    name: "ASSIST2012",
    usedInPaper: true,
    sourceLabel: "ASSISTments 2012-2013 School Data",
    sourceUrl:
      "https://sites.google.com/site/assistmentsdata/home/2012-13-school-data-with-affect",
    tag: "Paper dataset",
    nInteraction: "2711813",
    nUser: "29018",
    nConcept: "265",
    nQuestion: "53091",
    avgSeqLen: "96.41",
    avgAnswerAcc: "0.6954",
    questionSparsity: "0.9983",
    conceptSparsity: "0.9493",
    sideInfo: "concept_text, school_id, timestamp, use_time, num_hint, num_attempt",
  },
  {
    name: "STATICS2011",
    usedInPaper: true,
    sourceLabel: "PSLC DataShop: Statics 2011",
    sourceUrl: "https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=507",
    tag: "Paper dataset",
    nInteraction: "189297",
    nUser: "333",
    nConcept: "27",
    nQuestion: "1223",
    avgSeqLen: "568.46",
    avgAnswerAcc: "0.7654",
    questionSparsity: "0.5398",
    conceptSparsity: "0.3516",
    sideInfo: "concept_text, question_text, timestamp",
  },
  {
    name: "EdNet-KT1",
    usedInPaper: true,
    sourceLabel: "EdNet Official Repository",
    sourceUrl: "https://github.com/riiid/ednet",
    tag: "Paper dataset",
    nInteraction: "26056109",
    nUser: "5000",
    nConcept: "188",
    nQuestion: "12272",
    avgSeqLen: "5211.22",
    avgAnswerAcc: "0.702",
    questionSparsity: "0.6793",
    conceptSparsity: "0.08",
    sideInfo: "timestamp, use_time, num_hint, num_attempt",
  },
  {
    name: "ASSIST2012 (pyKT split variant)",
    usedInPaper: false,
    sourceLabel: "ASSISTments 2012-2013 School Data",
    sourceUrl:
      "https://sites.google.com/site/assistmentsdata/home/2012-13-school-data-with-affect",
    tag: "Additional supported dataset",
    nInteraction: "-",
    nUser: "-",
    nConcept: "-",
    nQuestion: "-",
    avgSeqLen: "-",
    avgAnswerAcc: "-",
    questionSparsity: "-",
    conceptSparsity: "-",
    sideInfo: "-",
  },
  {
    name: "Algebra 2005",
    usedInPaper: false,
    sourceLabel: "PSLC DataShop: Algebra 2005-2006",
    sourceUrl: "https://pslcdatashop.web.cmu.edu/",
    tag: "Additional supported dataset",
    nInteraction: "-",
    nUser: "-",
    nConcept: "-",
    nQuestion: "-",
    avgSeqLen: "-",
    avgAnswerAcc: "-",
    questionSparsity: "-",
    conceptSparsity: "-",
    sideInfo: "-",
  },
  {
    name: "Bridge to Algebra 2006",
    usedInPaper: false,
    sourceLabel: "PSLC DataShop: Bridge to Algebra 2006-2007",
    sourceUrl: "https://pslcdatashop.web.cmu.edu/",
    tag: "Additional supported dataset",
    nInteraction: "-",
    nUser: "-",
    nConcept: "-",
    nQuestion: "-",
    avgSeqLen: "-",
    avgAnswerAcc: "-",
    questionSparsity: "-",
    conceptSparsity: "-",
    sideInfo: "-",
  },
  {
    name: "Spanish",
    usedInPaper: false,
    sourceLabel: "Duolingo SLAM Shared Task",
    sourceUrl: "https://sharedtask.duolingo.com/",
    tag: "Additional supported dataset",
    nInteraction: "-",
    nUser: "-",
    nConcept: "-",
    nQuestion: "-",
    avgSeqLen: "-",
    avgAnswerAcc: "-",
    questionSparsity: "-",
    conceptSparsity: "-",
    sideInfo: "-",
  },
  {
    name: "STATICS (processed split variant)",
    usedInPaper: false,
    sourceLabel: "PSLC DataShop: Statics 2011",
    sourceUrl: "https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=507",
    tag: "Additional supported dataset",
    nInteraction: "-",
    nUser: "-",
    nConcept: "-",
    nQuestion: "-",
    avgSeqLen: "-",
    avgAnswerAcc: "-",
    questionSparsity: "-",
    conceptSparsity: "-",
    sideInfo: "-",
  },
];

const FIELD_MEANING = {
  used_in_paper: "Whether this dataset is used in the paper experiments.",
  name: "Dataset name and whether it is part of the paper benchmark.",
  n_interaction: "Total number of learner-question interactions.",
  n_user: "Total number of learners.",
  n_concept: "Total number of concepts or knowledge components.",
  n_question: "Total number of questions or items.",
  avg_seq_len: "Average interaction sequence length per learner.",
  avg_answer_acc: "Average answer accuracy over interactions.",
  question_sparsity: "Question-level sparsity statistic.",
  concept_sparsity: "Concept-level sparsity statistic.",
  side_info: "Available side information fields in the dataset.",
  link: "Official source page for this dataset.",
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
          <td>${item.nInteraction}</td>
          <td>${item.nUser}</td>
          <td>${item.nConcept}</td>
          <td>${item.nQuestion}</td>
          <td>${item.avgSeqLen}</td>
          <td>${item.avgAnswerAcc}</td>
          <td>${item.questionSparsity}</td>
          <td>${item.conceptSparsity}</td>
          <td class="left">${item.sideInfo}</td>
          <td><a href="${item.sourceUrl}" title="${item.sourceLabel}" target="_blank" rel="noreferrer">source</a></td>
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
