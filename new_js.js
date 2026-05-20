
    // ═══════════════════ NEW UI FEATURES ═══════════════════

    // ── AI Config card toggle ──
    document.getElementById("aiConfigToggle").addEventListener("click", () => {
      document.getElementById("aiConfigCard").classList.toggle("collapsed");
    });

    // ── API Key toggle ──
    document.getElementById("toggleApiKeyBtn").addEventListener("click", () => {
      const inp = document.getElementById("aiApiKey");
      inp.type = inp.type === "password" ? "text" : "password";
    });

    // ── Table search / filter / sort ──
    let allRecords = [];
    let activeFilter = "all";
    const tableSearch = document.getElementById("tableSearch");
    const tableFilter = document.getElementById("tableFilter");
    const tableCount = document.getElementById("tableCount");
    const clearFilterBtn = document.getElementById("clearFilterBtn");

    function applyTableFilters() {
      const query = (tableSearch.value || "").toLowerCase();
      const filter = tableFilter.value;
      const rows = document.querySelectorAll("#records tr");
      let visible = 0;
      rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        const statusClass = row.querySelector("td:nth-child(5)")?.className || "";
        let show = text.includes(query);
        if (filter === "pass") show = show && statusClass.includes("pass") && !statusClass.includes("review");
        else if (filter === "fail") show = show && statusClass.includes("fail");
        else if (filter === "review") show = show && (statusClass.includes("review") || statusClass.includes("unknown"));
        row.style.display = show ? "" : "none";
        if (show) visible++;
      });
      tableCount.textContent = visible + " / " + rows.length + " 条记录";
    }
    tableSearch.addEventListener("input", applyTableFilters);
    tableFilter.addEventListener("change", applyTableFilters);
    clearFilterBtn.addEventListener("click", () => {
      tableSearch.value = "";
      tableFilter.value = "all";
      activeFilter = "all";
      document.querySelectorAll(".dash-card").forEach(c => c.classList.remove("active-filter"));
      applyTableFilters();
    });

    // ── Dashboard card filtering ──
    document.querySelectorAll(".dash-card[data-filter]").forEach(card => {
      card.addEventListener("click", () => {
        const filter = card.dataset.filter;
        if (activeFilter === filter) {
          activeFilter = "all";
          card.classList.remove("active-filter");
          tableFilter.value = "all";
        } else {
          document.querySelectorAll(".dash-card").forEach(c => c.classList.remove("active-filter"));
          activeFilter = filter;
          card.classList.add("active-filter");
          tableFilter.value = filter === "review" ? "review" : filter;
        }
        applyTableFilters();
      });
    });

    // ── Override renderAnalyze for dashboard + downloads ──
    const origRenderAnalyze = renderAnalyze;
    renderAnalyze = function(result) {
      origRenderAnalyze(result);
      const stats = result.statistics || {};
      const judged = (stats.pass_count || 0) + (stats.fail_count || 0);
      const totalR = stats.total_records || 0;
      const passR = stats.pass_count || 0;
      const failR = stats.fail_count || 0;
      const reviewR = stats.review_count || 0;
      // Hero stats
      document.getElementById("hsReports").textContent = judged || "0";
      document.getElementById("hsItems").textContent = totalR;
      document.getElementById("hsAbnormal").textContent = failR;
      document.getElementById("hsRate").textContent = judged ? ((passR/judged)*100).toFixed(0)+"%" : "-";
      // Dashboard
      document.getElementById("dashPass").textContent = passR;
      document.getElementById("dashFormula").textContent = judged ? ((passR/judged)*100).toFixed(0)+"%" : "-";
      // Downloads
      const dl = result.downloads || {};
      document.getElementById("dlExcel").innerHTML = dl.Excel ? '<a href="'+dl.Excel+'" target="_blank">下载 Excel</a>' : '<span class="dl-status">等待生成</span>';
      document.getElementById("dlPdf").innerHTML = dl["增强 PDF"] ? '<a href="'+dl['增强 PDF']+'" target="_blank">下载 PDF</a>' : '<span class="dl-status">等待生成</span>';
      document.getElementById("dlHtml").innerHTML = dl.HTML ? '<a href="'+dl.HTML+'" target="_blank">下载 HTML</a>' : '<span class="dl-status">等待生成</span>';
      document.getElementById("dlJson").innerHTML = dl.JSON ? '<a href="'+dl.JSON+'" target="_blank">下载 JSON</a>' : '<span class="dl-status">等待生成</span>';
      applyTableFilters();
    };

    // ── File info display ──
    filesInput.addEventListener("change", () => {
      const info = document.getElementById("fileInfo");
      if (filesInput.files.length) {
        const items = [];
        let totalSize = 0;
        [...filesInput.files].forEach((f, i) => {
          const size = f.size < 1048576 ? (f.size/1024).toFixed(0)+" KB" : (f.size/1048576).toFixed(1)+" MB";
          totalSize += f.size;
          items.push(i+1+". "+f.name+' ('+size+", "+f.name.split(".").pop().toUpperCase()+")");
        });
        const total = totalSize < 1048576 ? (totalSize/1024).toFixed(0)+" KB" : (totalSize/1048576).toFixed(1)+" MB";
        info.innerHTML = "已选择 <b>"+filesInput.files.length+"</b> 个文件，共 <b>"+total+"</b><br>"+items.slice(0,8).join("<br>")+(items.length>8?"<br>…及其他 "+(items.length-8)+" 个文件":"");
        info.style.display = "block";
      } else {
        info.style.display = "none";
      }
    });

    // ── Step navigation click ──
    document.querySelectorAll(".step").forEach(el => {
      el.addEventListener("click", () => {
        const n = parseInt(el.id.replace("step",""));
        const isDone = el.classList.contains("done");
        const isActive = el.classList.contains("active");
        if (isDone || isActive) {
          const targets = [null, "#aiConfigCard", "#files", "#runBtn", "#verifyBtn", "#dashSection"];
          const target = targets[n];
          if (target) {
            const elTarget = document.querySelector(target);
            if (elTarget) elTarget.scrollIntoView({behavior:"smooth",block:"center"});
          }
        } else {
          statusEl.textContent = "请先完成前面的步骤";
        }
      });
    });
