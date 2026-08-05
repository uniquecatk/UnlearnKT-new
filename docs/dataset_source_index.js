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
    sourceUrl: "https://sites.google.com/view/assistmentsdatamining/dataset?authuser=0",
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
    name: "SLP-chi",
    usedInPaper: false,
    sourceLabel: "Dataset reference",
    sourceUrl: "https://aic-fe.bnu.edu.cn/en/data/index.html",
    tag: "Additional dataset reference",
    nInteraction: "80888",
    nUser: "623",
    nConcept: "31",
    nQuestion: "637",
    avgSeqLen: "129.84",
    avgAnswerAcc: "0.7494",
    questionSparsity: "0.7973",
    conceptSparsity: "0.4796",
    sideInfo: "timestamp, school_id, gender",
  },
  {
    name: "SLP-eng",
    usedInPaper: false,
    sourceLabel: "Dataset reference",
    sourceUrl: "https://aic-fe.bnu.edu.cn/en/data/index.html",
    tag: "Additional dataset reference",
    nInteraction: "86530",
    nUser: "366",
    nConcept: "28",
    nQuestion: "1089",
    avgSeqLen: "236.42",
    avgAnswerAcc: "0.784",
    questionSparsity: "0.7832",
    conceptSparsity: "0.1755",
    sideInfo: "timestamp, school_id, gender",
  },
  {
    name: "moocradar-C746997",
    usedInPaper: false,
    sourceLabel: "Dataset reference",
    sourceUrl: "https://github.com/THU-KEG/MOOC-Radar",
    tag: "Additional dataset reference",
    nInteraction: "100066",
    nUser: "1577",
    nConcept: "265",
    nQuestion: "550",
    avgSeqLen: "63.45",
    avgAnswerAcc: "0.6858",
    questionSparsity: "0.8846",
    conceptSparsity: "0.842",
    sideInfo: "concept_text, question_text, timestamp, cognitive_dimension",
  },
  {
    name: "SLP-phy",
    usedInPaper: false,
    sourceLabel: "Dataset reference",
    sourceUrl: "https://aic-fe.bnu.edu.cn/en/data/index.html",
    tag: "Additional dataset reference",
    nInteraction: "107288",
    nUser: "664",
    nConcept: "54",
    nQuestion: "1915",
    avgSeqLen: "161.58",
    avgAnswerAcc: "0.6122",
    questionSparsity: "0.9159",
    conceptSparsity: "0.341",
    sideInfo: "timestamp, school_id, gender",
  },
  {
    name: "SLP-geo",
    usedInPaper: false,
    sourceLabel: "Dataset reference",
    sourceUrl: "https://aic-fe.bnu.edu.cn/en/data/index.html",
    tag: "Additional dataset reference",
    nInteraction: "149780",
    nUser: "1135",
    nConcept: "47",
    nQuestion: "1011",
    avgSeqLen: "132.08",
    avgAnswerAcc: "0.6316",
    questionSparsity: "0.8694",
    conceptSparsity: "0.379",
    sideInfo: "timestamp, school_id, gender",
  },
  {
    name: "DBE-KT22",
    usedInPaper: false,
    sourceLabel: "Dataset reference",
    sourceUrl: "https://dataverse.ada.edu.au/dataset.xhtml?persistentId=doi:10.26193/6DZWOH",
    tag: "Additional dataset reference",
    nInteraction: "158342",
    nUser: "1214",
    nConcept: "93",
    nQuestion: "212",
    avgSeqLen: "130.43",
    avgAnswerAcc: "0.7646",
    questionSparsity: "0.3861",
    conceptSparsity: "0.3534",
    sideInfo: "timestamp, concept_text, question_text",
  },
  {
    name: "SLP-mat",
    usedInPaper: false,
    sourceLabel: "Dataset reference",
    sourceUrl: "https://aic-fe.bnu.edu.cn/en/data/index.html",
    tag: "Additional dataset reference",
    nInteraction: "242722",
    nUser: "1499",
    nConcept: "44",
    nQuestion: "1127",
    avgSeqLen: "161.92",
    avgAnswerAcc: "0.6761",
    questionSparsity: "0.8565",
    conceptSparsity: "0.1896",
    sideInfo: "timestamp, school_id, gender",
  },
  {
    name: "SLP-bio",
    usedInPaper: false,
    sourceLabel: "Dataset reference",
    sourceUrl: "https://aic-fe.bnu.edu.cn/en/data/index.html",
    tag: "Additional dataset reference",
    nInteraction: "291800",
    nUser: "1941",
    nConcept: "23",
    nQuestion: "1058",
    avgSeqLen: "150.33",
    avgAnswerAcc: "0.6575",
    questionSparsity: "0.858",
    conceptSparsity: "0.1795",
    sideInfo: "timestamp, school_id, gender",
  },
  {
    name: "SLP-his",
    usedInPaper: false,
    sourceLabel: "Dataset reference",
    sourceUrl: "https://aic-fe.bnu.edu.cn/en/data/index.html",
    tag: "Additional dataset reference",
    nInteraction: "296711",
    nUser: "1610",
    nConcept: "22",
    nQuestion: "1251",
    avgSeqLen: "184.29",
    avgAnswerAcc: "0.7278",
    questionSparsity: "0.8532",
    conceptSparsity: "0.4535",
    sideInfo: "timestamp, school_id, gender",
  },
  {
    name: "Assist2009-full",
    usedInPaper: false,
    sourceLabel: "Dataset reference",
    sourceUrl: "https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/combined-dataset-2009-10)",
    tag: "Additional dataset reference",
    nInteraction: "432672",
    nUser: "6593",
    nConcept: "151",
    nQuestion: "13544",
    avgSeqLen: "65.63",
    avgAnswerAcc: "0.6308",
    questionSparsity: "0.9954",
    conceptSparsity: "0.8976",
    sideInfo: "-",
  },
  {
    name: "Poj",
    usedInPaper: false,
    sourceLabel: "Dataset reference",
    sourceUrl: "https://github.com/riiid/ednet",
    tag: "Additional dataset reference",
    nInteraction: "996240",
    nUser: "22916",
    nConcept: "2750",
    nQuestion: "2750",
    avgSeqLen: "50.75",
    avgAnswerAcc: "0.3552",
    questionSparsity: "0.9975",
    conceptSparsity: "0.9975",
    sideInfo: "timestamp, answer_error_type",
  },
  {
    name: "Slepemapy-anatomy",
    usedInPaper: false,
    sourceLabel: "Dataset reference",
    sourceUrl: "http://data.practiceanatomy.com/",
    tag: "Additional dataset reference",
    nInteraction: "1173566",
    nUser: "18540",
    nConcept: "246",
    nQuestion: "5730",
    avgSeqLen: "76.64",
    avgAnswerAcc: "0.7491",
    questionSparsity: "0.9893",
    conceptSparsity: "0.9279",
    sideInfo: "concept_text, timestamp, use_time",
  },
  {
    name: "Edi2020-task34",
    usedInPaper: false,
    sourceLabel: "Dataset reference",
    sourceUrl: "https://eedi.com/projects/neurips-education-challenge",
    tag: "Additional dataset reference",
    nInteraction: "1382727",
    nUser: "4918",
    nConcept: "53",
    nQuestion: "948",
    avgSeqLen: "281.16",
    avgAnswerAcc: "0.5373",
    questionSparsity: "0.7034",
    conceptSparsity: "0.519",
    sideInfo: "concept_text, question_text, timestamp, user_age",
  },
  {
    name: "Edi2020-task1-longest-seqs-5000",
    usedInPaper: false,
    sourceLabel: "Dataset reference",
    sourceUrl: "https://eedi.com/projects/neurips-education-challenge",
    tag: "Additional dataset reference",
    nInteraction: "3568804",
    nUser: "5000",
    nConcept: "282",
    nQuestion: "27613",
    avgSeqLen: "713.76",
    avgAnswerAcc: "0.628",
    questionSparsity: "0.9742",
    conceptSparsity: "0.7194",
    sideInfo: "concept_text, timestamp, user_age",
  },
  {
    name: "Xes3g5m",
    usedInPaper: false,
    sourceLabel: "Dataset reference",
    sourceUrl: "https://github.com/ai4ed/XES3G5M",
    tag: "Additional dataset reference",
    nInteraction: "5549184",
    nUser: "18066",
    nConcept: "865",
    nQuestion: "7652",
    avgSeqLen: "307.16",
    avgAnswerAcc: "0.7947",
    questionSparsity: "0.9616",
    conceptSparsity: "0.8362",
    sideInfo: "concept_text, question_text, timestamp",
  },
  {
    name: "Edi2020-task1",
    usedInPaper: false,
    sourceLabel: "Dataset reference",
    sourceUrl: "https://eedi.com/projects/neurips-education-challenge",
    tag: "Additional dataset reference",
    nInteraction: "19834813",
    nUser: "118971",
    nConcept: "282",
    nQuestion: "27613",
    avgSeqLen: "166.72",
    avgAnswerAcc: "0.643",
    questionSparsity: "0.994",
    conceptSparsity: "0.8882",
    sideInfo: "concept_text, timestamp, user_age",
  },
  {
    name: "Ednet-kt1-longest-seqs-5000",
    usedInPaper: false,
    sourceLabel: "Dataset reference",
    sourceUrl: "https://github.com/riiid/ednet",
    tag: "Additional dataset reference",
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
