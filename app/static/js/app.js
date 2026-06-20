/**
 * Explainable Medical AI - Frontend SPA Orchestration
 */

document.addEventListener("DOMContentLoaded", () => {
    // Global State
    let currentResultData = null;
    let compareDataA = null;
    let compareDataB = null;
    let distributionChart = null;
    
    // Select DOM Elements
    const body = document.body;
    const themeToggleBtn = document.getElementById("themeToggleBtn");
    const themeIcon = document.getElementById("themeIcon");
    const predictionLoader = document.getElementById("predictionLoader");
    const statusToast = document.getElementById("statusToast");
    const toastTitle = document.getElementById("toastTitle");
    const toastBody = document.getElementById("toastBody");
    const bsToast = new bootstrap.Toast(statusToast);

    // Initializations
    initTheme();
    initTabs();
    initUpload();
    initHistory();
    initCompare();

    /* ==========================================================================
       THEME MANAGEMENT (Dark / Light Mode)
       ========================================================================== */
    function initTheme() {
        const savedTheme = localStorage.getItem("medical-ai-theme");
        if (savedTheme === "dark") {
            body.classList.add("dark-mode");
            themeIcon.className = "bi bi-sun-fill text-warning";
        } else {
            body.classList.remove("dark-mode");
            themeIcon.className = "bi bi-moon-fill";
        }

        themeToggleBtn.addEventListener("click", () => {
            body.classList.toggle("dark-mode");
            const isDark = body.classList.contains("dark-mode");
            localStorage.setItem("medical-ai-theme", isDark ? "dark" : "light");
            themeIcon.className = isDark ? "bi bi-sun-fill text-warning" : "bi bi-moon-fill";
            
            // Re-render chart if it exists to match theme styling
            if (distributionChart) {
                renderDistributionChart(
                    parseInt(document.getElementById("statPneumoniaCases").textContent),
                    parseInt(document.getElementById("statTotalScans").textContent) - parseInt(document.getElementById("statPneumoniaCases").textContent)
                );
            }
        });
    }

    /* ==========================================================================
       TAB NAVIGATION
       ========================================================================== */
    function initTabs() {
        const navLinks = document.querySelectorAll(".nav-link-custom");
        const tabSections = document.querySelectorAll(".tab-content");

        navLinks.forEach(link => {
            link.addEventListener("click", (e) => {
                e.preventDefault();
                const targetId = link.getAttribute("data-tab-target");
                
                // Update nav classes
                navLinks.forEach(l => l.classList.remove("active"));
                link.classList.add("active");
                
                // Update visible sections
                tabSections.forEach(section => {
                    if (section.id === `tab-${targetId}`) {
                        section.classList.remove("d-none");
                    } else {
                        section.classList.add("d-none");
                    }
                });

                // Refresh history or stats if navigating to history tab
                if (targetId === "history-stats") {
                    fetchHistory();
                }
            });
        });
    }

    /* ==========================================================================
       TOAST NOTIFICATIONS UTILITY
       ========================================================================== */
    function showToast(title, message, isError = false) {
        toastTitle.textContent = title;
        toastBody.textContent = message;
        if (isError) {
            statusToast.classList.add("text-danger");
        } else {
            statusToast.classList.remove("text-danger");
        }
        bsToast.show();
    }

    /* ==========================================================================
       FILE UPLOAD AND VALIDATION
       ========================================================================== */
    function initUpload() {
        const dropzone = document.getElementById("dropzone");
        const fileInput = document.getElementById("xrayFileInput");
        const patientInput = document.getElementById("patientNameInput");
        const runPredictBtn = document.getElementById("runPredictBtn");
        const previewBox = document.getElementById("uploadPreviewBox");
        const previewName = document.getElementById("previewFileName");
        const previewSize = document.getElementById("previewFileSize");
        const clearPreviewBtn = document.getElementById("clearPreviewBtn");

        let selectedFile = null;

        // Click on dropzone triggers input click
        dropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropzone.classList.add("border-primary");
        });

        dropzone.addEventListener("dragleave", () => {
            dropzone.classList.remove("border-primary");
        });

        dropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropzone.classList.remove("border-primary");
            if (e.dataTransfer.files.length > 0) {
                handleFileSelection(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener("change", (e) => {
            if (e.target.files.length > 0) {
                handleFileSelection(e.target.files[0]);
            }
        });

        function handleFileSelection(file) {
            const allowed = ["image/png", "image/jpeg", "image/jpg"];
            if (!allowed.includes(file.type)) {
                showToast("Invalid File Type", "Please upload a valid PNG, JPG, or JPEG image.", true);
                return;
            }
            
            // Check size limit (16MB)
            if (file.size > 16 * 1024 * 1024) {
                showToast("File Too Large", "Image size must be smaller than 16 Megabytes.", true);
                return;
            }

            selectedFile = file;
            previewName.textContent = file.name;
            previewSize.textContent = formatBytes(file.size);
            
            previewBox.classList.remove("d-none");
            runPredictBtn.removeAttribute("disabled");
        }

        clearPreviewBtn.addEventListener("click", () => {
            selectedFile = null;
            fileInput.value = "";
            previewBox.classList.add("d-none");
            runPredictBtn.setAttribute("disabled", "true");
        });

        // Trigger Diagnosis
        runPredictBtn.addEventListener("click", () => {
            if (!selectedFile) return;
            
            predictionLoader.style.display = "flex";
            
            const formData = new FormData();
            formData.append("image", selectedFile);
            formData.append("patient_name", patientInput.value.trim());

            fetch("/predict", {
                method: "POST",
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => { throw new Error(err.error || "Server error"); });
                }
                return response.json();
            })
            .then(data => {
                currentResultData = data;
                populateDiagnosticResults(data);
                showToast("Analysis Complete", `Scan for patient ${data.patient_name} successfully diagnosed.`);
            })
            .catch(error => {
                console.error("Diagnosis error:", error);
                showToast("Diagnostic Failure", error.message, true);
            })
            .finally(() => {
                predictionLoader.style.display = "none";
            });
        });

        // Reset Diagnostic panel button
        document.getElementById("resetDiagBtn").addEventListener("click", () => {
            selectedFile = null;
            fileInput.value = "";
            patientInput.value = "";
            previewBox.classList.add("d-none");
            runPredictBtn.setAttribute("disabled", "true");
            
            document.getElementById("resultBox").classList.add("d-none");
            document.getElementById("emptyResultBox").classList.remove("d-none");
            currentResultData = null;
        });
    }

    // Helper to format file sizes
    function formatBytes(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    /* ==========================================================================
       DIAGNOSTIC DISPLAY POPULATION (Dashboard Output Panel)
       ========================================================================== */
    function populateDiagnosticResults(data) {
        // Toggle view panels
        document.getElementById("emptyResultBox").classList.add("d-none");
        const resultBox = document.getElementById("resultBox");
        resultBox.classList.remove("d-none");
        resultBox.scrollIntoView({ behavior: "smooth" });

        // Status pill & label
        const predPill = document.getElementById("resultPredictionPill");
        const predLabel = document.getElementById("resultPredictionLabel");
        
        predPill.textContent = data.prediction.toUpperCase();
        predLabel.textContent = data.prediction;
        
        if (data.prediction === "Pneumonia") {
            predPill.className = "status-pill status-pneumonia";
            predPill.innerHTML = '<i class="bi bi-exclamation-triangle-fill"></i> Pneumonia';
        } else {
            predPill.className = "status-pill status-normal";
            predPill.innerHTML = '<i class="bi bi-patch-check-fill"></i> Normal';
        }

        // Processing latency
        document.getElementById("resultLatencyText").textContent = `${data.processing_time.toFixed(2)} seconds`;

        // Update Circular Confidence Gauge
        const confPercent = data.confidence;
        document.getElementById("resultConfidenceText").textContent = `${confPercent.toFixed(1)}%`;
        
        const circle = document.getElementById("gaugeProgressBar");
        // Circumference of our SVG circle (radius 70) is 2 * PI * 70 = ~439.8
        const circumference = 2 * Math.PI * 70;
        const offset = circumference - (confPercent / 100) * circumference;
        circle.style.strokeDasharray = circumference;
        circle.style.strokeDashoffset = offset;
        
        // Gauge color matches predicted state
        circle.style.stroke = data.prediction === "Pneumonia" ? "var(--accent)" : "var(--success)";

        // Probability bars
        const pneumProb = data.confidence_pneumonia;
        const normProb = data.confidence_normal;
        
        document.getElementById("pneumoniaProbText").textContent = `${pneumProb.toFixed(1)}%`;
        document.getElementById("pneumoniaProgressBar").style.width = `${pneumProb}%`;
        
        document.getElementById("normalProbText").textContent = `${normProb.toFixed(1)}%`;
        document.getElementById("normalProgressBar").style.width = `${normProb}%`;

        // Qualitative findings & explanations checklist
        const checklist = document.getElementById("findingsChecklistContainer");
        checklist.innerHTML = "";
        
        let checklistItems = [];
        let summaryText = "";
        
        if (data.prediction === "Pneumonia") {
            checklistItems = [
                "Lung Opacity and consolidations detected in thoracic lobes",
                "Lower lobe densities indicating local inflammatory fluid accumulation",
                "Increased pulmonary vascular markings and opacity spreads"
            ];
            summaryText = "The AI detected visual markers of density, opacity, and lobar infiltrates, which strongly increases the probability of pneumonia.";
        } else {
            checklistItems = [
                "Healthy dark lung fields indicating uniform aeration",
                "Absence of consolidated tissue or active infections",
                "Clear vascular structures and normal costophrenic angles"
            ];
            summaryText = "The scan presents healthy aerated lung tissue, without abnormal focal consolidations, leading to a classification of Normal.";
        }

        checklistItems.forEach(finding => {
            const item = document.createElement("div");
            item.className = "finding-list-item";
            
            const checkIcon = document.createElement("div");
            checkIcon.className = `finding-check-icon ${data.prediction === "Pneumonia" ? "finding-pneumonia-check" : "finding-normal-check"}`;
            checkIcon.innerHTML = "✓";
            
            const text = document.createElement("span");
            text.className = "small fw-medium";
            text.textContent = finding;
            
            item.appendChild(checkIcon);
            item.appendChild(text);
            checklist.appendChild(item);
        });

        document.getElementById("xaiExplanationSummaryText").textContent = summaryText;

        // Image View controllers setup
        const mainImg = document.getElementById("mainResultImg");
        const dloadBtn = document.getElementById("downloadCurrentOverlayBtn");
        
        // Set original upload path as base source
        mainImg.src = `/static/uploads/${data.filename}`;
        dloadBtn.href = `/static/uploads/${data.filename}`;
        
        // Reset explanation tabs
        const viewerButtons = document.querySelectorAll("[data-view]");
        viewerButtons.forEach(btn => btn.classList.remove("active"));
        document.getElementById("viewImageBtn").classList.add("active");

        // Document PDF downloader route binding
        document.getElementById("downloadPdfBtn").href = `/report/${data.id}`;
    }

    // Toggle XAI Explanation Images (Original, Grad-CAM, SHAP, LIME)
    const viewerButtons = document.querySelectorAll("[data-view]");
    viewerButtons.forEach(button => {
        button.addEventListener("click", () => {
            if (!currentResultData) return;
            
            // Toggle active classes
            viewerButtons.forEach(btn => btn.classList.remove("active"));
            button.classList.add("active");
            
            const viewType = button.getAttribute("data-view");
            const mainImg = document.getElementById("mainResultImg");
            const dloadBtn = document.getElementById("downloadCurrentOverlayBtn");
            
            let targetSrc = "";
            if (viewType === "original") {
                targetSrc = `/static/uploads/${currentResultData.filename}`;
            } else if (viewType === "gradcam") {
                targetSrc = currentResultData.heatmap_path;
            } else if (viewType === "shap") {
                targetSrc = currentResultData.shap_path;
            } else if (viewType === "lime") {
                targetSrc = currentResultData.lime_path;
            }
            
            mainImg.src = targetSrc;
            dloadBtn.href = targetSrc;
        });
    });

    /* ==========================================================================
       COMPARATIVE клинический VIEW
       ========================================================================== */
    function initCompare() {
        setupCompareSlot("A");
        setupCompareSlot("B");
        
        function setupCompareSlot(slotId) {
            const dropzone = document.getElementById(`compareDropzone${slotId}`);
            const input = document.getElementById(`compareFileInput${slotId}`);
            const viewer = document.getElementById(`compareViewer${slotId}`);
            const img = document.getElementById(`compareImg${slotId}`);
            const info = document.getElementById(`compareInfo${slotId}`);
            const clear = document.getElementById(`compareClear${slotId}`);
            const overlaySelect = document.getElementById(`compareOverlay${slotId}`);
            
            dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("border-primary"); });
            dropzone.addEventListener("dragleave", () => dropzone.classList.remove("border-primary"));
            dropzone.addEventListener("drop", (e) => {
                e.preventDefault();
                dropzone.classList.remove("border-primary");
                if (e.dataTransfer.files.length > 0) {
                    processCompareFile(e.dataTransfer.files[0]);
                }
            });
            
            input.addEventListener("change", (e) => {
                if (e.target.files.length > 0) {
                    processCompareFile(e.target.files[0]);
                }
            });
            
            function processCompareFile(file) {
                predictionLoader.style.display = "flex";
                
                const formData = new FormData();
                formData.append("image", file);
                formData.append("patient_name", `Compare Scan ${slotId}`);
                
                fetch("/predict", {
                    method: "POST",
                    body: formData
                })
                .then(res => {
                    if(!res.ok) { return res.json().then(e => { throw new Error(e.error); }); }
                    return res.json();
                })
                .then(data => {
                    if (slotId === "A") compareDataA = data;
                    else compareDataB = data;
                    
                    // Show viewer
                    dropzone.classList.add("d-none");
                    viewer.classList.remove("d-none");
                    info.classList.remove("d-none");
                    
                    // Set image
                    img.src = `/static/uploads/${data.filename}`;
                    overlaySelect.value = "orig";
                    
                    // Set diagnosis label
                    const resLabel = document.getElementById(`compareResult${slotId}`);
                    const confLabel = document.getElementById(`compareConfidence${slotId}`);
                    
                    resLabel.textContent = data.prediction;
                    resLabel.className = `status-pill ${data.prediction === 'Pneumonia' ? 'status-pneumonia' : 'status-normal'}`;
                    confLabel.textContent = `Confidence: ${data.confidence.toFixed(1)}%`;
                    
                    showToast(`Scan ${slotId} Loaded`, `Diagnosis completed for Slot ${slotId}.`);
                })
                .catch(err => {
                    showToast(`Slot ${slotId} Error`, err.message, true);
                })
                .finally(() => {
                    predictionLoader.style.display = "none";
                });
            }
            
            // Dropdown Overlay select changer
            overlaySelect.addEventListener("change", () => {
                const data = slotId === "A" ? compareDataA : compareDataB;
                if (!data) return;
                
                const selected = overlaySelect.value;
                img.src = selected === "orig" ? `/static/uploads/${data.filename}` : data.heatmap_path;
            });
            
            // Clear viewer
            clear.addEventListener("click", () => {
                if (slotId === "A") compareDataA = null;
                else compareDataB = null;
                
                input.value = "";
                dropzone.classList.remove("d-none");
                viewer.classList.add("d-none");
                info.classList.add("d-none");
            });
        }
    }

    /* ==========================================================================
       HISTORY AND STATISTICS LOGIC
       ========================================================================== */
    function initHistory() {
        const searchInput = document.getElementById("historySearchInput");
        
        // Search table records filter
        searchInput.addEventListener("input", (e) => {
            const query = e.target.value.toLowerCase().trim();
            const rows = document.querySelectorAll("#historyTableBody tr");
            
            rows.forEach(row => {
                // If it's the 'no records' row, ignore
                if (row.cells.length === 1) return;
                
                const name = row.cells[0].textContent.toLowerCase();
                if (name.includes(query)) {
                    row.classList.remove("d-none");
                } else {
                    row.classList.add("d-none");
                }
            });
        });
    }

    function fetchHistory() {
        fetch("/history")
        .then(response => response.json())
        .then(data => {
            populateHistoryTable(data);
            populateStats(data);
        })
        .catch(error => {
            console.error("Error retrieving historical logs:", error);
            showToast("Database Error", "Failed to retrieve diagnostic records.", true);
        });
    }

    function populateHistoryTable(records) {
        const tbody = document.getElementById("historyTableBody");
        tbody.innerHTML = "";
        
        if (records.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="5" class="text-center py-5 text-secondary">
                        <i class="bi bi-database-exclamation display-6 d-block mb-2"></i>
                        No diagnostic logs found in local database.
                    </td>
                </tr>
            `;
            return;
        }

        records.forEach(record => {
            const tr = document.createElement("tr");
            
            // Prediction badge formatting
            const isPneumonia = record.prediction === "Pneumonia";
            const badgeClass = isPneumonia ? "status-pneumonia" : "status-normal";
            const icon = isPneumonia ? "bi-exclamation-triangle-fill" : "bi-patch-check-fill";

            tr.innerHTML = `
                <td class="ps-4 fw-bold text-primary-light">${record.patient_name}</td>
                <td>
                    <span class="status-pill ${badgeClass}" style="padding: 0.25rem 0.65rem; font-size: 0.75rem;">
                        <i class="bi ${icon}"></i> ${record.prediction}
                    </span>
                </td>
                <td class="fw-bold">${record.confidence.toFixed(1)}%</td>
                <td class="text-secondary small">${record.date}</td>
                <td class="pe-4 text-end">
                    <button class="btn btn-sm btn-outline-primary me-1 view-record-btn" data-record-id="${record.id}" title="Load details in dashboard">
                        <i class="bi bi-box-arrow-in-right"></i> View
                    </button>
                    <a href="/report/${record.id}" class="btn btn-sm btn-outline-secondary me-1" title="Download PDF report">
                        <i class="bi bi-file-earmark-pdf"></i> PDF
                    </a>
                    <button class="btn btn-sm btn-outline-danger delete-record-btn" data-record-id="${record.id}" title="Delete record">
                        <i class="bi bi-trash"></i>
                    </button>
                </td>
            `;

            // View event listener
            tr.querySelector(".view-record-btn").addEventListener("click", () => {
                // Populate dashboard result box
                currentResultData = record;
                populateDiagnosticResults(record);
                
                // Navigate to dashboard tab
                document.querySelector("[data-tab-target='dashboard']").click();
            });

            // Delete event listener
            tr.querySelector(".delete-record-btn").addEventListener("click", () => {
                if (confirm(`Are you sure you want to delete patient record for ${record.patient_name}? This will permanently delete database entries and files.`)) {
                    deleteRecord(record.id);
                }
            });

            tbody.appendChild(tr);
        });
    }

    function deleteRecord(id) {
        fetch(`/history/${id}`, {
            method: "DELETE"
        })
        .then(res => {
            if(!res.ok) throw new Error("Delete failed");
            return res.json();
        })
        .then(() => {
            showToast("Record Deleted", `Successfully deleted diagnostic record #${id}.`);
            fetchHistory(); // Reload history and statistics
        })
        .catch(err => {
            showToast("Deletion Error", err.message, true);
        });
    }

    /* ==========================================================================
       ANALYTICS & CHART.JS GENERATION
       ========================================================================== */
    function populateStats(records) {
        const total = records.length;
        document.getElementById("statTotalScans").textContent = total;
        
        if (total === 0) {
            document.getElementById("statPneumoniaCases").textContent = 0;
            document.getElementById("statPneumoniaRate").textContent = "0% of total scans";
            document.getElementById("statAvgConfidence").textContent = "0%";
            document.getElementById("statAvgLatency").textContent = "0.0s";
            
            if (distributionChart) {
                distributionChart.destroy();
                distributionChart = null;
            }
            return;
        }

        const pneumoniaCases = records.filter(r => r.prediction === "Pneumonia").length;
        const pneumoniaRate = (pneumoniaCases / total) * 100;
        
        let confidenceSum = 0;
        let latencySum = 0;
        records.forEach(r => {
            confidenceSum += r.confidence;
            latencySum += r.processing_time;
        });

        const avgConfidence = confidenceSum / total;
        const avgLatency = latencySum / total;

        document.getElementById("statPneumoniaCases").textContent = pneumoniaCases;
        document.getElementById("statPneumoniaRate").textContent = `${pneumoniaRate.toFixed(1)}% of total scans`;
        document.getElementById("statAvgConfidence").textContent = `${avgConfidence.toFixed(1)}%`;
        document.getElementById("statAvgLatency").textContent = `${avgLatency.toFixed(1)}s`;

        // Render doughnut distribution chart
        const normalCases = total - pneumoniaCases;
        renderDistributionChart(pneumoniaCases, normalCases);
    }

    function renderDistributionChart(pneumonia, normal) {
        const ctx = document.getElementById("distributionChart").getContext("2d");
        
        // Destruct previous chart if it exists
        if (distributionChart) {
            distributionChart.destroy();
        }

        const isDarkMode = body.classList.contains("dark-mode");
        const textCol = isDarkMode ? "#e0e0e0" : "#1e293b";

        distributionChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Pneumonia', 'Normal'],
                datasets: [{
                    data: [pneumonia, normal],
                    backgroundColor: ['#E64A19', '#2E7D32'],
                    borderColor: isDarkMode ? '#1e293b' : '#ffffff',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: textCol,
                            font: { family: 'Inter', weight: 600 }
                        }
                    }
                }
            }
        });
    }
});
