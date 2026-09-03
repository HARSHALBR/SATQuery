
        let t1Data = null;
        let t2Data = null;
        let map = null;
        let mapRect = null;

        // Initialize map
        document.addEventListener('DOMContentLoaded', () => {
            map = L.map('map', {zoomControl: false, attributionControl: false}).setView([0, 0], 2);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                maxZoom: 19
            }).addTo(map);
        });

        const demoScenarios = {
            'veg_increase': {
                name: 'Vegetation Increased (Wildfire Recovery)',
                query: 'Has vegetation increased in this region?',
                t1: 'Frontend/data/demo_scenarios/veg_increase/t1/t1',
                t2: 'Frontend/data/demo_scenarios/veg_increase/t2/t2',
                bands: ['red', 'nir', 'scl'],
                metadata: {crs: 'EPSG:32610', width: 100, height: 100, dev_scenario: 'NORMAL'} // -> SUPPORTED
            },
            'builtup_decrease': {
                name: 'Built-up Area Decreased (Demolition)',
                query: 'Has the built-up area decreased?',
                t1: 'Frontend/data/demo_scenarios/builtup_decrease/t1/t1',
                t2: 'Frontend/data/demo_scenarios/builtup_decrease/t2/t2',
                bands: ['red', 'nir', 'swir', 'scl'],
                metadata: {crs: 'EPSG:32610', width: 100, height: 100, dev_scenario: 'CONFLICTING_EVIDENCE'} // -> UNCERTAIN
            },
            'flood_change': {
                name: 'Water Body Change (Flooding)',
                query: 'How has the water body changed over time?',
                t1: 'Frontend/data/demo_scenarios/flood_change/t1/t1',
                t2: 'Frontend/data/demo_scenarios/flood_change/t2/t2',
                bands: ['red', 'nir', 'scl'],
                metadata: {crs: 'EPSG:32610', width: 100, height: 100, dev_scenario: 'LOW_QUALITY'} // -> INSUFFICIENT
            }
        };

        function selectScenario(scenarioId) {
            const data = demoScenarios[scenarioId];
            document.getElementById('queryInput').value = data.query;
            
            t1Data = {
                status: 'success',
                observation_id: `t1_${scenarioId}`,
                image_path: data.t1,
                bands: data.bands,
                metadata: data.metadata
            };
            t2Data = {
                status: 'success',
                observation_id: `t2_${scenarioId}`,
                image_path: data.t2,
                bands: data.bands,
                metadata: data.metadata
            };
            
            document.getElementById('t1Status').classList.add('hidden');
            document.getElementById('t2Status').classList.add('hidden');
            
            const valBox = document.getElementById('validationResult');
            valBox.classList.remove('hidden', 'bg-red-50', 'text-red-700', 'border-red-200');
            valBox.classList.add('bg-emerald-50', 'text-emerald-700', 'border-emerald-200');
            valBox.innerHTML = `<div class="flex items-center gap-3">
                <span class="text-2xl">✓</span>
                <strong class="block">DEMO FIXTURE LOADED</strong>
            </div>`;
            
            document.getElementById('analyzeBtn').disabled = false;
            document.getElementById('analyzeBtn').classList.remove('opacity-50', 'cursor-not-allowed');
        }

        async function uploadFile(obsRole) {
            const fileInput = document.getElementById(`${obsRole}File`);
            const statusBox = document.getElementById(`${obsRole}Status`);
            
            if (!fileInput.files.length) {
                alert("Please select a ZIP file first.");
                return;
            }
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            statusBox.classList.remove('hidden', 'bg-red-50', 'text-red-700');
            statusBox.classList.add('bg-blue-50', 'text-blue-700');
            statusBox.innerHTML = `<span class="flex items-center gap-2"><div class="loader" style="width:14px;height:14px;border-width:2px;"></div> Uploading & Validating...</span>`;

            try {
                const response = await fetch('/api/v1/upload', { method: 'POST', body: formData });
                const data = await response.json();
                
                if (data.status === 'success') {
                    statusBox.classList.remove('bg-blue-50', 'text-blue-700');
                    statusBox.classList.add('bg-emerald-50', 'text-emerald-700', 'border-emerald-200');
                    statusBox.innerHTML = `✓ Validated: ${data.bands.join(', ')}<br><span class="text-[10px] font-mono break-all">${data.image_path}</span>`;
                    if (obsRole === 't1') t1Data = data;
                    if (obsRole === 't2') t2Data = data;
                } else {
                    statusBox.classList.remove('bg-blue-50', 'text-blue-700');
                    statusBox.classList.add('bg-red-50', 'text-red-700', 'border-red-200');
                    statusBox.innerHTML = `🔴 Error: ${data.message}`;
                    if (obsRole === 't1') t1Data = null;
                    if (obsRole === 't2') t2Data = null;
                }
            } catch (err) {
                statusBox.classList.remove('bg-blue-50', 'text-blue-700');
                statusBox.classList.add('bg-red-50', 'text-red-700', 'border-red-200');
                statusBox.innerHTML = `🔴 Network Error: ${err.message}`;
            }
            checkReadiness();
        }

        function checkReadiness() {
            const btn = document.getElementById('analyzeBtn');
            const valBox = document.getElementById('validationResult');
            
            if (t1Data && t1Data.status === 'success' && t2Data && t2Data.status === 'success') {
                if (t1Data.metadata.crs !== t2Data.metadata.crs) {
                    valBox.classList.remove('hidden', 'bg-emerald-50', 'text-emerald-700', 'border-emerald-200');
                    valBox.classList.add('bg-red-50', 'text-red-700', 'border-red-200');
                    valBox.innerHTML = `<strong>🔴 INVALID INPUT:</strong> Spatial metadata mismatch.`;
                    btn.disabled = true;
                    btn.classList.add('opacity-50', 'cursor-not-allowed');
                } else {
                    valBox.classList.remove('hidden', 'bg-red-50', 'text-red-700', 'border-red-200');
                    valBox.classList.add('bg-emerald-50', 'text-emerald-700', 'border-emerald-200');
                    valBox.innerHTML = `<strong>✓ VALID INPUT:</strong> Observations successfully validated and aligned.`;
                    btn.disabled = false;
                    btn.classList.remove('opacity-50', 'cursor-not-allowed');
                }
            }
        }

        async function runAnalysis() {
            if (!t1Data || !t2Data) return;
            
            const btn = document.getElementById('analyzeBtn');
            const query = document.getElementById('queryInput').value;
            
            document.getElementById('resultsContainer').classList.add('hidden');
            document.getElementById('loadingState').classList.remove('hidden');
            document.getElementById('loadingState').classList.add('flex');
            
            btn.disabled = true;
            btn.classList.add('opacity-50', 'cursor-not-allowed');

            const payload = {
                query: query || "Analyze changes in the region.",
                observations: [
                    { observation_id: t1Data.observation_id, image_path: t1Data.image_path, role: "t1", metadata: { modality: "optical", bands: t1Data.bands } },
                    { observation_id: t2Data.observation_id, image_path: t2Data.image_path, role: "t2", metadata: { modality: "optical", bands: t2Data.bands } }
                ],
                metadata: { dev_scenario: t1Data.metadata.dev_scenario } 
            };

            try {
                const response = await fetch('/api/v1/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await response.json();
                renderResults(data);
            } catch (err) {
                console.error(err);
                alert("Analysis failed to connect to API.");
            } finally {
                btn.disabled = false;
                btn.classList.remove('opacity-50', 'cursor-not-allowed');
                document.getElementById('loadingState').classList.add('hidden');
                document.getElementById('loadingState').classList.remove('flex');
            }
        }

        function renderResults(data) {
            document.getElementById('resultsContainer').classList.remove('hidden');
            
            // 01 Query
            var el = document.getElementById('uiTask'); if (el) el.textContent = (data.task || "Unknown Task").replace('_', ' ');
            let targetText = "Region";
            if (data.task === 'vegetation_change') targetText = "🌿 Vegetation";
            if (data.task === 'built_up_change') targetText = "🏗️ Built-up Area";
            if (data.task === 'flood_change') targetText = "💧 Water Body";
            var el = document.getElementById('uiTarget'); if (el) el.textContent = targetText;

            // 02 Validation
            if(t1Data && t1Data.metadata) {
                var el = document.getElementById('uiRes'); if (el) el.textContent = `${t1Data.metadata.width || 10980} × ${t1Data.metadata.height || 10980} px • ${t1Data.metadata.crs || 'EPSG:32633'}`;
                document.getElementById('uiBands').innerHTML = t1Data.bands.map(b => `${b.toUpperCase()} <span class="text-emerald-500">✓</span>`).join(' &nbsp; ');
            }

            // 03 Planner Tools
            const uiPlannerRS = document.getElementById('uiPlannerRS');
            uiPlannerRS.innerHTML = '';
            let reqEv = ['RS-InternVL'];
            if (data.execution_trace) {
                const executed = new Set();
                data.execution_trace.forEach(t => {
                    if (t.tool !== "validate_inputs" && t.tool !== "compare_evidence" && t.tool !== "generate_response" && t.tool !== "run_rs_vlm") {
                        executed.add(t.tool);
                    }
                });
                
                executed.forEach(tool => {
                    reqEv.push(tool.replace('_', ' ').toUpperCase());
                    let desc = "Quantitative analysis";
                    if (tool.includes('ndvi')) desc = "Vegetation index change";
                    if (tool.includes('ndbi')) desc = "Built-up index change";
                    if (tool.includes('sar')) desc = "Independent radar evidence";
                    if (tool.includes('statistics')) desc = "Magnitude + affected area";

                    uiPlannerRS.innerHTML += `
                        <div class="flex items-start gap-2">
                            <span class="text-emerald-400">✓</span>
                            <div>
                                <span class="text-slate-200 capitalize">${tool.replace('_', ' ')}</span>
                                <span class="block text-xs text-slate-400 font-normal">${desc}</span>
                            </div>
                        </div>
                    `;
                });
                if(executed.size === 0) uiPlannerRS.innerHTML = `<span class="text-xs text-slate-500 italic">None executed</span>`;
            }
            var el = document.getElementById('uiRequiredEv'); if (el) el.textContent = reqEv.join(' • ');

            // 04 Multi-Path Evidence
            const pA = document.getElementById('uiPathA');
            const pB = document.getElementById('uiPathB');
            pA.innerHTML = ''; pB.innerHTML = '';
            
            let vlmEv = false;
            let rsEv = false;
            let boundingBox = null;
            let summaryMetrics = [];
            let evidenceCount = 0;

            if (data.evidence && data.evidence.length > 0) {
                // Find VLM Interpretation
                const vlm = data.evidence.find(e => e.type === 'vlm_interpretation');
                if (vlm && vlm.value) {
                    vlmEv = true;
                    evidenceCount++;
                    const val = vlm.value;
                    const semanticQuery = val.semantic_query || "Not available";
                    const t1Claim = (val.t1 && val.t1.claim) || val.claim || "Not available";
                    const t1Reason = (val.t1 && val.t1.reasoning) || "Not available";
                    const t1Conf = (val.t1 && val.t1.confidence) || "Not available";
                    const t2Claim = (val.t2 && val.t2.claim) || val.claim || "Not available";
                    const t2Reason = (val.t2 && val.t2.reasoning) || val.reasoning || "Not available";
                    const t2Conf = (val.t2 && val.t2.confidence) || val.confidence || "Not available";
                    
                    const score = (val.t2 && val.t2.confidence) || val.confidence || "Not available";
                    const scoreText = score !== "Not available" ? (score * 100).toFixed(0) + '%' : "Not available";
                    const t1ConfText = t1Conf !== "Not available" ? (t1Conf * 100).toFixed(0) + '%' : "Not available";
                    const t2ConfText = t2Conf !== "Not available" ? (t2Conf * 100).toFixed(0) + '%' : "Not available";
                    
                    const qualityStat = (vlm.quality && vlm.quality.valid_pixel_fraction !== undefined) ? (vlm.quality.valid_pixel_fraction > 0.5 ? "VALID" : "LOW QUALITY") : "Not available";
                    const vlmProvenanceTool = vlm.provenance ? vlm.provenance.tool : "run_rs_vlm";
                    const vlmIsMock = vlm.evidence_id && vlm.evidence_id.includes('mock');
                    const vlmDemoBadge = vlmIsMock ? `<span class="text-[8px] bg-amber-500/20 text-amber-300 px-1.5 py-0.5 rounded border border-amber-500/50 uppercase ml-2 tracking-widest font-bold">DEMO MODE: Simulated Evidence</span>` : '';

                    pA.innerHTML = `
                        <div class="mb-4 p-3 bg-purple-900/30 rounded-lg border border-purple-800/50">
                            <span class="text-[10px] text-purple-400 font-bold uppercase block mb-1">Semantic Query</span>
                            <span class="text-sm text-white font-medium">"${semanticQuery}"</span>
                        </div>
                        <div class="bg-slate-900/80 rounded-lg border border-purple-700/50 p-4 shadow-sm mb-3">
                            <div class="flex justify-between items-center mb-3 pb-2 border-b border-purple-900/50">
                                <span class="text-xs font-bold text-slate-300">T1 / Before</span>
                                <span class="text-[10px] font-mono bg-purple-900/50 px-1.5 py-0.5 rounded text-purple-300 border border-purple-700/50">Conf: ${t1ConfText}</span>
                            </div>
                            <div class="mb-2">
                                <span class="text-[10px] text-purple-400 font-bold uppercase block mb-0.5">Model Claim</span>
                                <span class="text-sm text-white">${t1Claim}</span>
                            </div>
                            <p class="text-xs text-slate-400 italic border-l-2 border-purple-700/50 pl-2">"${t1Reason}"</p>
                        </div>
                        <div class="bg-slate-900/80 rounded-lg border border-purple-700/50 p-4 shadow-sm mb-4">
                            <div class="flex justify-between items-center mb-3 pb-2 border-b border-purple-900/50">
                                <span class="text-xs font-bold text-slate-300">T2 / After</span>
                                <span class="text-[10px] font-mono bg-purple-900/50 px-1.5 py-0.5 rounded text-purple-300 border border-purple-700/50">Conf: ${t2ConfText}</span>
                            </div>
                            <div class="mb-2">
                                <span class="text-[10px] text-purple-400 font-bold uppercase block mb-0.5">Model Claim</span>
                                <span class="text-sm text-white">${t2Claim}</span>
                            </div>
                            <p class="text-xs text-slate-400 italic border-l-2 border-purple-700/50 pl-2">"${t2Reason}"</p>
                        </div>
                        <div class="bg-slate-900/50 rounded-lg border border-slate-700/50 p-3 mt-4">
                            <h4 class="text-[9px] font-bold text-slate-400 uppercase tracking-widest mb-2 border-b border-slate-700/50 pb-1 flex items-center">Evidence Provenance ${vlmDemoBadge}</h4>
                            <div class="grid grid-cols-2 gap-3 text-[10px] font-mono text-slate-300">
                                <div><span class="text-slate-500 block mb-0.5">SOURCE</span>${vlmProvenanceTool}</div>
                                <div><span class="text-slate-500 block mb-0.5">QUALITY</span><span class="${qualityStat==='VALID'?'text-emerald-400':'text-amber-400'}">${qualityStat}</span></div>
                                <div class="col-span-2"><span class="text-slate-500 block mb-0.5">EVIDENCE ID</span><span class="text-purple-400">${vlm.evidence_id || 'Not available'}</span></div>
                            </div>
                        </div>
                    `;
                    summaryMetrics.push(`VLM: ${t2Claim} (${scoreText} conf)`);
                }

                // Find RS Quantitative Evidence
                const rsEvidences = data.evidence.filter(e => e.type !== 'vlm_interpretation' && e.type !== 'spatial_grounding');
                if (rsEvidences.length > 0) {
                    rsEv = true;
                    rsEvidences.forEach(ev => {
                        evidenceCount++;
                        const val = ev.value || {};
                        let displayVal = "";
                        let summaryStr = "";
                        let toolName = ev.type.replace('_', ' ').toUpperCase();
                        
                        // Fallbacks to handle exactly the keys available
                        const formatVal = (v) => v !== undefined ? v : "Not available";
                        const formatPct = (v) => v !== undefined ? (v*100).toFixed(1) + '%' : "Not available";
                        const formatFix = (v) => v !== undefined ? (typeof v === 'number' ? v.toFixed(3) : v) : "Not available";

                        if (ev.type === 'vegetation_change') {
                            toolName = "NDVI DELTA";
                            displayVal = `
                                ${val.ndvi_before !== undefined ? `<div class="flex justify-between mb-1.5"><span class="text-slate-400">NDVI Before:</span> <span class="font-bold text-white">${formatFix(val.ndvi_before)}</span></div>` : ''}
                                ${val.ndvi_after !== undefined ? `<div class="flex justify-between mb-1.5"><span class="text-slate-400">NDVI After:</span> <span class="font-bold text-white">${formatFix(val.ndvi_after)}</span></div>` : ''}
                                <div class="flex justify-between mb-1.5"><span class="text-slate-400">NDVI Δ:</span> <span class="font-bold text-white">${formatFix(val.ndvi_delta)}</span></div>
                                <div class="flex justify-between mb-1.5"><span class="text-slate-400">Interpretation:</span> <span class="font-bold uppercase ${val.direction==='decrease'?'text-red-400':'text-emerald-400'}">${formatVal(val.direction)}</span></div>
                                <div class="mt-3 pt-2 border-t border-emerald-900/50">
                                    <div class="flex justify-between mb-1.5"><span class="text-slate-400">Changed Pixels:</span> <span class="font-bold text-white">${formatPct(val.affected_fraction)}</span></div>
                                </div>
                            `;
                            summaryStr = `NDVI: Δ ${formatFix(val.ndvi_delta)}`;
                        } else if (ev.type === 'built_up_change') {
                            toolName = "NDBI DELTA";
                            displayVal = `
                                ${val.ndbi_before !== undefined ? `<div class="flex justify-between mb-1.5"><span class="text-slate-400">NDBI Before:</span> <span class="font-bold text-white">${formatFix(val.ndbi_before)}</span></div>` : ''}
                                ${val.ndbi_after !== undefined ? `<div class="flex justify-between mb-1.5"><span class="text-slate-400">NDBI After:</span> <span class="font-bold text-white">${formatFix(val.ndbi_after)}</span></div>` : ''}
                                <div class="flex justify-between mb-1.5"><span class="text-slate-400">NDBI Δ:</span> <span class="font-bold text-white">${formatFix(val.ndbi_delta)}</span></div>
                                <div class="flex justify-between mb-1.5"><span class="text-slate-400">Interpretation:</span> <span class="font-bold uppercase ${val.direction==='decrease'?'text-red-400':'text-emerald-400'}">${formatVal(val.direction)}</span></div>
                                <div class="mt-3 pt-2 border-t border-emerald-900/50">
                                    <div class="flex justify-between mb-1.5"><span class="text-slate-400">Affected Area:</span> <span class="font-bold text-white">${formatPct(val.affected_fraction)}</span></div>
                                </div>
                            `;
                            summaryStr = `NDBI: Δ ${formatFix(val.ndbi_delta)}`;
                        } else if (ev.type === 'sar_amplitude_change') {
                            toolName = "SAR CHANGE";
                            displayVal = `
                                ${val.vv_delta !== undefined ? `<div class="flex justify-between mb-1.5"><span class="text-slate-400">VV Δ:</span> <span class="font-bold text-white">${formatFix(val.vv_delta)}</span></div>` : ''}
                                ${val.vh_delta !== undefined ? `<div class="flex justify-between mb-1.5"><span class="text-slate-400">VH Δ:</span> <span class="font-bold text-white">${formatFix(val.vh_delta)}</span></div>` : ''}
                                <div class="flex justify-between mb-1.5"><span class="text-slate-400">Change Detected:</span> <span class="font-bold text-white">${val.sar_change_detected ? 'Yes' : 'No'}</span></div>
                                <div class="mt-3 pt-2 border-t border-emerald-900/50">
                                    <div class="flex justify-between mb-1.5"><span class="text-slate-400">Change Score:</span> <span class="font-bold text-white">${formatFix(val.change_score)}</span></div>
                                </div>
                            `;
                            summaryStr = `SAR: Score ${formatFix(val.change_score)}`;
                        } else if (ev.type === 'change_quantification') {
                            toolName = "CHANGE STATISTICS";
                            displayVal = `
                                <div class="flex justify-between mb-1.5"><span class="text-slate-400">Change Detected:</span> <span class="font-bold text-white">${val.change_detected ? 'Yes' : 'No'}</span></div>
                                <div class="flex justify-between mb-1.5"><span class="text-slate-400">Changed Pixels:</span> <span class="font-bold text-white">${formatPct(val.changed_pixel_fraction)}</span></div>
                            `;
                            summaryStr = `Stats: ${formatPct(val.changed_pixel_fraction)}`;
                        } else {
                            Object.keys(val).forEach(k => {
                                displayVal += `<div class="flex justify-between mb-1.5"><span class="text-slate-400 capitalize">${k.replace(/_/g, ' ')}:</span> <span class="font-bold text-white">${val[k]}</span></div>`;
                            });
                        }
                        
                        const qualityStat = (ev.quality && ev.quality.valid_pixel_fraction !== undefined) ? (ev.quality.valid_pixel_fraction > 0.5 ? "VALID" : "LOW QUALITY") : "Not available";
                        const rsProvenanceTool = ev.provenance ? ev.provenance.tool : ev.type;
                        const rsIsMock = ev.evidence_id && ev.evidence_id.includes('mock');
                        const rsDemoBadge = rsIsMock ? `<span class="text-[8px] bg-amber-500/20 text-amber-300 px-1.5 py-0.5 rounded border border-amber-500/50 uppercase ml-2 tracking-widest font-bold">DEMO MODE: Simulated Evidence</span>` : '';

                        pB.innerHTML += `
                            <div class="bg-slate-900/80 rounded-lg border border-emerald-700/50 p-4 shadow-sm mb-4 text-sm">
                                <h4 class="text-[10px] font-bold text-emerald-400 uppercase tracking-widest mb-3 border-b border-emerald-900/50 pb-2">${toolName}</h4>
                                <div class="mb-4">${displayVal}</div>
                                <div class="bg-slate-900/50 rounded-lg border border-slate-700/50 p-3 mt-4">
                                    <h4 class="text-[9px] font-bold text-slate-400 uppercase tracking-widest mb-2 border-b border-slate-700/50 pb-1 flex items-center">Evidence Provenance ${rsDemoBadge}</h4>
                                    <div class="grid grid-cols-2 gap-3 text-[10px] font-mono text-slate-300">
                                        <div><span class="text-slate-500 block mb-0.5">SOURCE</span>${rsProvenanceTool}</div>
                                        <div><span class="text-slate-500 block mb-0.5">QUALITY</span><span class="${qualityStat==='VALID'?'text-emerald-400':'text-amber-400'}">${qualityStat}</span></div>
                                        <div class="col-span-2"><span class="text-slate-500 block mb-0.5">EVIDENCE ID</span><span class="text-emerald-400">${ev.evidence_id || 'Not available'}</span></div>
                                    </div>
                                </div>
                            </div>
                        `;
                        if(summaryStr) summaryMetrics.push(summaryStr);
                    });
                }

                // Map Bounding Box
                const groundEv = data.evidence.find(e => e.type === 'spatial_grounding');
                if (groundEv && groundEv.value && groundEv.value.bounding_box) {
                    boundingBox = groundEv.value.bounding_box;
                }
            }

            if(!vlmEv) pA.innerHTML = '<div class="text-slate-400 text-sm font-mono text-center p-8 bg-slate-900/50 rounded-xl border border-slate-700/50">Not available<br><span class="text-xs text-slate-500 mt-2 block">No semantic evidence returned for this analysis.</span></div>';
            if(!rsEv) pB.innerHTML = '<div class="text-slate-400 text-sm font-mono text-center p-8 bg-slate-900/50 rounded-xl border border-slate-700/50">Not available<br><span class="text-xs text-slate-500 mt-2 block">No quantitative evidence returned for this analysis.</span></div>';

            // 05 Comparator
            const compCard = document.getElementById('comparatorCard');
            const compBadge = document.getElementById('comparatorBadge');
            const compReason = document.getElementById('comparatorReason');
            const reportCard = document.getElementById('reportCard');
            
            compCard.className = 'dark-glass-card p-0 overflow-hidden mx-auto max-w-4xl shadow-xl border-2 mb-12';
            reportCard.className = 'dark-glass-card p-0 flex flex-col overflow-hidden border-l-8';
            
            if (data.status === 'SUPPORTED') {
                compCard.classList.add('border-emerald-500');
                compBadge.className = 'text-5xl md:text-6xl font-black tracking-tighter mb-6 text-emerald-500 drop-shadow-md';
                compBadge.innerHTML = '✓ SUPPORTED';
                reportCard.classList.add('border-emerald-500');
            } else if (data.status === 'UNCERTAIN' || data.status === 'CONFLICTING') {
                compCard.classList.add('border-amber-500');
                compBadge.className = 'text-5xl md:text-6xl font-black tracking-tighter mb-6 text-amber-500 drop-shadow-md';
                compBadge.innerHTML = '⚠ UNCERTAIN';
                reportCard.classList.add('border-amber-500');
            } else {
                compCard.classList.add('border-red-500');
                compBadge.className = 'text-5xl md:text-6xl font-black tracking-tighter mb-6 text-red-500 drop-shadow-md';
                compBadge.innerHTML = '❌ INSUFFICIENT';
                reportCard.classList.add('border-red-500');
            }
            
            // Extract reason from limits or answer if not explicitly provided as reason string
            let reasonText = "No reason provided by comparator.";
            if (data.reason) {
                reasonText = data.reason;
            } else if (data.limitations && data.limitations.length > 0 && data.status !== 'SUPPORTED') {
                reasonText = data.limitations[0];
            } else if (data.answer) {
                const parts = data.answer.split('Reason: ');
                if (parts.length > 1) {
                    reasonText = parts[1].split('\n')[0];
                }
            }
            if (compReason) compReason.textContent = reasonText;

            // Comparator Evidence Summary Tree
            const uiCompDetails = document.getElementById('uiComparatorDetails');
            uiCompDetails.innerHTML = '';
            
            const vlmSummary = data.vlm_summary || (vlmEv ? "Analysis complete" : "Not executed");
            const rsSummary = data.rs_summary || (rsEv ? "Measurements returned" : "Not executed");
            const compReasonText = data.reason || reasonText;
            
            if (data.status === 'SUPPORTED') {
                uiCompDetails.innerHTML = `
                    <div class="grid grid-cols-2 gap-8 relative max-w-2xl mx-auto">
                        <div class="text-center p-4 bg-emerald-900/10 border border-emerald-900/30 rounded-xl relative">
                            <span class="text-[10px] font-bold text-slate-400 uppercase tracking-widest block mb-2">VLM Conclusion</span>
                            <span class="text-white font-bold block">${vlmSummary}</span>
                            <div class="absolute -bottom-6 left-1/2 w-0.5 h-6 bg-emerald-500/50"></div>
                        </div>
                        <div class="text-center p-4 bg-emerald-900/10 border border-emerald-900/30 rounded-xl relative">
                            <span class="text-[10px] font-bold text-slate-400 uppercase tracking-widest block mb-2">RS Conclusion</span>
                            <span class="text-white font-bold block">${rsSummary}</span>
                            <div class="absolute -bottom-6 left-1/2 w-0.5 h-6 bg-emerald-500/50"></div>
                        </div>
                    </div>
                    <div class="flex justify-center mt-6 max-w-2xl mx-auto">
                        <div class="w-1/2 border-t-2 border-emerald-500/50 relative">
                            <div class="absolute left-1/2 -top-3 -ml-10 w-20 text-center bg-emerald-500 text-black text-[10px] font-bold px-2 py-0.5 rounded">AGREEMENT</div>
                            <div class="absolute left-1/2 top-0 w-0.5 h-6 bg-emerald-500/50"></div>
                        </div>
                    </div>
                `;
                compBadge.className = 'text-5xl md:text-6xl font-black tracking-tighter mb-4 text-emerald-500 drop-shadow-md';
                compBadge.innerHTML = 'SUPPORTED';
                document.getElementById('comparatorReasonTitle').innerHTML = 'Evidence Agreement';
            } else if (data.status === 'UNCERTAIN' || data.status === 'CONFLICTING') {
                uiCompDetails.innerHTML = `
                    <div class="grid grid-cols-2 gap-8 relative max-w-2xl mx-auto">
                        <div class="text-center p-4 bg-amber-900/10 border border-amber-900/30 rounded-xl relative">
                            <span class="text-[10px] font-bold text-slate-400 uppercase tracking-widest block mb-2">VLM Conclusion</span>
                            <span class="text-white font-bold block">${vlmSummary}</span>
                            <div class="absolute -bottom-6 left-1/2 w-0.5 h-6 bg-amber-500/50"></div>
                        </div>
                        <div class="text-center p-4 bg-amber-900/10 border border-amber-900/30 rounded-xl relative">
                            <span class="text-[10px] font-bold text-slate-400 uppercase tracking-widest block mb-2">RS Conclusion</span>
                            <span class="text-white font-bold block">${rsSummary}</span>
                            <div class="absolute -bottom-6 left-1/2 w-0.5 h-6 bg-amber-500/50"></div>
                        </div>
                    </div>
                    <div class="flex justify-center mt-6 max-w-2xl mx-auto">
                        <div class="w-1/2 border-t-2 border-amber-500/50 border-dashed relative">
                            <div class="absolute left-1/2 -top-3 -ml-10 w-20 text-center bg-amber-500 text-black text-[10px] font-bold px-2 py-0.5 rounded">CONFLICT</div>
                            <div class="absolute left-1/2 top-0 w-0.5 h-6 bg-amber-500/50"></div>
                        </div>
                    </div>
                `;
                compBadge.className = 'text-5xl md:text-6xl font-black tracking-tighter mb-4 text-amber-500 drop-shadow-md';
                compBadge.innerHTML = 'UNCERTAIN';
                document.getElementById('comparatorReasonTitle').innerHTML = 'Evidence Conflict / Weak Agreement';
            } else {
                uiCompDetails.innerHTML = `
                    <div class="grid grid-cols-2 gap-8 relative max-w-2xl mx-auto">
                        <div class="text-center p-4 bg-red-900/10 border border-red-900/30 rounded-xl relative">
                            <span class="text-[10px] font-bold text-slate-400 uppercase tracking-widest block mb-2">VLM Path</span>
                            <span class="text-white font-bold block">${vlmSummary}</span>
                            <div class="absolute -bottom-6 left-1/2 w-0.5 h-6 bg-red-500/50"></div>
                        </div>
                        <div class="text-center p-4 bg-red-900/10 border border-red-900/30 rounded-xl relative">
                            <span class="text-[10px] font-bold text-slate-400 uppercase tracking-widest block mb-2">RS Path</span>
                            <span class="text-white font-bold block">${rsSummary}</span>
                            <div class="absolute -bottom-6 left-1/2 w-0.5 h-6 bg-red-500/50"></div>
                        </div>
                    </div>
                    <div class="flex justify-center mt-6 max-w-2xl mx-auto">
                        <div class="w-1/2 border-t-2 border-red-500/50 border-dotted relative">
                            <div class="absolute left-1/2 -top-3 -ml-12 w-24 text-center bg-red-500 text-black text-[10px] font-bold px-2 py-0.5 rounded">MISSING</div>
                            <div class="absolute left-1/2 top-0 w-0.5 h-6 bg-red-500/50"></div>
                        </div>
                    </div>
                `;
                compBadge.className = 'text-5xl md:text-6xl font-black tracking-tighter mb-4 text-red-500 drop-shadow-md';
                compBadge.innerHTML = 'INSUFFICIENT';
                document.getElementById('comparatorReasonTitle').innerHTML = 'Evidence Missing';
            }
            if (compReason) compReason.textContent = compReasonText;

            // 06 Final Report Summary
            let cleanAnswer = data.answer || "No answer generated.";
            if (cleanAnswer.includes("Reason:")) {
                cleanAnswer = cleanAnswer.split("Reason:")[0].trim();
            }
            document.getElementById('finalAnswer').innerHTML = cleanAnswer.replace(/\n/g, '<br>');
            var el = document.getElementById('uiFinalQuery'); if (el) el.textContent = document.getElementById('queryInput').value;

            // Final Decision Header
            const uiFinalHeader = document.getElementById('uiFinalHeader');
            let headerColor = data.status === 'SUPPORTED' ? 'text-emerald-500' : (data.status === 'UNCERTAIN' || data.status === 'CONFLICTING' ? 'text-amber-500' : 'text-red-500');
            let headerBg = data.status === 'SUPPORTED' ? 'bg-emerald-500/10 border-emerald-500/30' : (data.status === 'UNCERTAIN' || data.status === 'CONFLICTING' ? 'bg-amber-500/10 border-amber-500/30' : 'bg-red-500/10 border-red-500/30');
            
            let explanation = '';
            if (data.status === 'SUPPORTED') explanation = "Independent evidence paths agree.";
            else if (data.status === 'UNCERTAIN' || data.status === 'CONFLICTING') explanation = "Independent evidence paths disagree or the quantitative evidence is weak.";
            else explanation = "Insufficient valid evidence to verify the claim.";

            uiFinalHeader.innerHTML = `
                <div class="inline-block p-4 rounded-xl border backdrop-blur-sm shadow-xl ${headerBg}">
                    <h3 class="text-2xl font-black tracking-tighter mb-1 ${headerColor}">${data.status}</h3>
                    <p class="text-sm text-slate-300">${explanation}</p>
                </div>
            `;

            // Border color for report card
            reportCard.className = 'dark-glass-card p-0 flex flex-col overflow-hidden border-t-4';
            if(data.status === 'SUPPORTED') reportCard.classList.add('border-t-emerald-500');
            else if(data.status === 'UNCERTAIN' || data.status === 'CONFLICTING') reportCard.classList.add('border-t-amber-500');
            else reportCard.classList.add('border-t-red-500');

            // Confidence
            const uiConfBadge = document.getElementById('uiConfidenceBadge');
            const vlmNode = (data.evidence || []).find(e => e.type === 'vlm_interpretation');
            if (vlmNode && vlmNode.value && vlmNode.value.t2 && vlmNode.value.t2.confidence) {
                uiConfBadge.classList.remove('hidden');
                uiConfBadge.innerHTML = `<span class="text-slate-400">Confidence:</span> <span class="text-white font-bold">${(vlmNode.value.t2.confidence * 100).toFixed(0)}%</span>`;
            } else {
                uiConfBadge.classList.add('hidden');
            }

            // Evidence Ledger
            const uiEvidenceLedger = document.getElementById('uiEvidenceLedger');
            uiEvidenceLedger.innerHTML = '';
            
            const formatPct = (v) => v !== undefined && typeof v === 'number' ? (v*100).toFixed(1) + '%' : "Not available";
            const formatFix = (v) => v !== undefined ? (typeof v === 'number' ? v.toFixed(3) : v) : "Not available";

            if (data.evidence && data.evidence.length > 0) {
                data.evidence.forEach(ev => {
                    const isMock = ev.evidence_id && ev.evidence_id.includes('mock');
                    const demoBadge = isMock ? `<span class="bg-amber-500/20 text-amber-300 px-1.5 py-0.5 rounded border border-amber-500/50 uppercase tracking-widest text-[8px] font-bold absolute top-2 right-2">DEMO MODE: Simulated</span>` : '';
                    
                    const q = (ev.quality && ev.quality.valid_pixel_fraction !== undefined) ? (ev.quality.valid_pixel_fraction > 0.5 ? "VALID" : "LOW QUALITY") : "N/A";
                    const qColor = q === 'VALID' ? 'text-emerald-400' : (q === 'LOW QUALITY' ? 'text-amber-400' : 'text-slate-400');
                    
                    let impRes = "";
                    let iconColor = "bg-slate-700 text-slate-300 border-slate-600";
                    if (ev.type === 'vlm_interpretation') {
                        impRes = (ev.value && ev.value.t2) ? ev.value.t2.claim : "Interpretation complete";
                        iconColor = "bg-purple-900/50 text-purple-400 border border-purple-700/50";
                    } else if (ev.type === 'vegetation_change') {
                        impRes = `Δ NDVI: ${formatFix(ev.value ? ev.value.ndvi_delta : undefined)}`;
                        iconColor = "bg-emerald-900/50 text-emerald-400 border border-emerald-700/50";
                    } else if (ev.type === 'built_up_change') {
                        impRes = `Δ NDBI: ${formatFix(ev.value ? ev.value.ndbi_delta : undefined)}`;
                        iconColor = "bg-emerald-900/50 text-emerald-400 border border-emerald-700/50";
                    } else if (ev.type === 'sar_amplitude_change') {
                        impRes = (ev.value && ev.value.sar_change_detected) ? "SAR Change Detected" : "No SAR Change";
                        iconColor = "bg-emerald-900/50 text-emerald-400 border border-emerald-700/50";
                    } else if (ev.type === 'change_quantification') {
                        impRes = `Changed pixels: ${formatPct(ev.value ? ev.value.changed_pixel_fraction : undefined)}`;
                        iconColor = "bg-emerald-900/50 text-emerald-400 border border-emerald-700/50";
                    } else {
                        impRes = "Data extracted";
                    }
                    
                    if (ev.type !== 'spatial_grounding') {
                        uiEvidenceLedger.innerHTML += `
                            <div class="relative bg-slate-900/50 border border-slate-700/50 rounded p-3 pt-4 flex items-start gap-3">
                                ${demoBadge}
                                <div class="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 text-xs font-black border ${iconColor}">${ev.type.charAt(0).toUpperCase()}</div>
                                <div class="flex-grow min-w-0">
                                    <div class="flex justify-between items-baseline mb-0.5">
                                        <span class="text-xs font-bold text-slate-200 truncate pr-2">${ev.provenance ? ev.provenance.tool : ev.type}</span>
                                        <span class="text-[9px] font-bold uppercase tracking-widest ${qColor} flex-shrink-0">${q}</span>
                                    </div>
                                    <p class="text-[11px] text-slate-300 mb-1 font-medium truncate">${impRes}</p>
                                    <p class="text-[9px] text-slate-500 font-mono truncate" title="${ev.evidence_id}">${ev.evidence_id}</p>
                                </div>
                            </div>
                        `;
                    }
                });
            } else {
                uiEvidenceLedger.innerHTML = `<div class="text-slate-500 italic text-sm text-center py-4 bg-slate-900/30 rounded border border-slate-800">No quantitative or semantic evidence returned.</div>`;
            }

            const limBox = document.getElementById('limitationsBox');
            if (data.limitations && data.limitations.length > 0 && data.status !== 'SUPPORTED') {
                limBox.classList.remove('hidden');
                var el = document.getElementById('finalLimitations'); if (el) el.textContent = data.limitations.join(' | ');
            } else {
                limBox.classList.add('hidden');
            }

            var el = document.getElementById('uiSummaryCount'); if (el) el.textContent = data.evidence ? data.evidence.length : 0;
            var el = document.getElementById('uiTraceId'); if (el) el.textContent = data.trace_id || "N/A";
            
            const uiTraceList = document.getElementById('uiTraceList');
            uiTraceList.innerHTML = '';
            if (data.evidence) {
                data.evidence.forEach(ev => {
                    uiTraceList.innerHTML += `<div class="truncate"><span class="text-slate-400">${ev.evidence_id}</span> <span class="text-slate-600">(${ev.type})</span></div>`;
                });
            }

            // Map — driven by data.spatial_evidence from API
            setTimeout(() => { map.invalidateSize(); }, 150);

            const se = data.spatial_evidence;
            const spatialBadge = document.getElementById('spatialStatusBadge');

            // Clear any previous layers
            if (mapRect) { map.removeLayer(mapRect); mapRect = null; }
            if (window.mapMarker) { map.removeLayer(window.mapMarker); window.mapMarker = null; }

            if (se && se.available && se.bounds_wgs84) {
                const b = se.bounds_wgs84;
                const leafletBounds = [[b.south, b.west], [b.north, b.east]];

                // Status-aware rectangle color
                let rectColor = '#3b82f6';
                let badgeText = 'ANALYZED REGION';
                let badgeCls = 'text-blue-400 border-blue-500/50 bg-blue-900/30';
                let popupLabel = 'Analyzed Region — T1 → T2';

                if (data.status === 'SUPPORTED') {
                    rectColor = '#10b981';
                    badgeText = 'VERIFIED ✓';
                    badgeCls = 'text-emerald-400 border-emerald-500/50 bg-emerald-900/30';
                    popupLabel = 'Verified Change Region — T1 → T2';
                } else if (data.status === 'UNCERTAIN' || data.status === 'CONFLICTING') {
                    rectColor = '#f59e0b';
                    badgeText = 'CONFLICTING';
                    badgeCls = 'text-amber-400 border-amber-500/50 bg-amber-900/30';
                    popupLabel = 'Change Analyzed — Evidence Conflicting';
                } else if (data.status === 'INSUFFICIENT') {
                    rectColor = '#ef4444';
                    badgeText = 'UNVERIFIED';
                    badgeCls = 'text-red-400 border-red-500/50 bg-red-900/30';
                    popupLabel = 'Spatial Extent Available — Change Could Not Be Verified';
                }

                // Draw rectangle
                mapRect = L.rectangle(leafletBounds, {
                    color: rectColor, weight: 2, fillOpacity: 0.12,
                    dashArray: data.status === 'INSUFFICIENT' ? '6, 4' : null
                }).addTo(map);

                // Center marker
                if (se.center) {
                    const markerIcon = L.divIcon({
                        className: '',
                        html: `<div style="width:10px;height:10px;border-radius:50%;background:${rectColor};border:2px solid white;box-shadow:0 0 0 2px ${rectColor}55;"></div>`,
                        iconAnchor: [5, 5]
                    });
                    window.mapMarker = L.marker([se.center.lat, se.center.lon], {icon: markerIcon})
                        .addTo(map)
                        .bindPopup(`<div style="font-family:monospace;font-size:11px;min-width:180px">
                            <strong>SATQuery AI Analysis Region</strong><br>
                            ${popupLabel}<br><br>
                            <span style="color:#94a3b8">CRS:</span> ${se.crs || 'EPSG:4326'}<br>
                            <span style="color:#94a3b8">Center:</span> ${se.center.lat.toFixed(5)}°, ${se.center.lon.toFixed(5)}°
                        </div>`);
                }

                map.fitBounds(leafletBounds, {padding: [30, 30], maxZoom: 14});

                // Show badge
                spatialBadge.className = `text-[9px] font-bold uppercase tracking-widest px-2 py-1 rounded border ${badgeCls}`;
                spatialBadge.textContent = badgeText;
                spatialBadge.classList.remove('hidden');

                // Hide placeholder
                document.getElementById('mapPlaceholder').style.display = 'none';

                // Populate metadata panel
                const ndviFmt = (v) => v !== undefined && typeof v === 'number' ? v.toFixed(3) : 'Not available';
                const pctFmt = (v) => v !== undefined && typeof v === 'number' ? (v * 100).toFixed(1) + '%' : 'Not available';

                // Find region label from spatial_grounding evidence
                const sgEv = (data.evidence || []).find(e => e.type === 'spatial_grounding');
                const regionLabel = (sgEv && sgEv.value && sgEv.value.label) ? sgEv.value.label : 'Not available';
                document.getElementById('spatialLabel').textContent = regionLabel;
                document.getElementById('spatialLat').textContent = `${b.south.toFixed(4)}° – ${b.north.toFixed(4)}°`;
                document.getElementById('spatialLon').textContent = `${b.west.toFixed(4)}° – ${b.east.toFixed(4)}°`;
                document.getElementById('spatialCRS').textContent = se.crs || 'EPSG:4326';

                // T1 / T2 observation IDs from query state
                document.getElementById('spatialT1').textContent = (t1Data && t1Data.observation_id) ? t1Data.observation_id : 'Not available';
                document.getElementById('spatialT2').textContent = (t2Data && t2Data.observation_id) ? t2Data.observation_id : 'Not available';

                // Change metrics from evidence
                const vegEv = (data.evidence || []).find(e => e.type === 'vegetation_change');
                const statsEv = (data.evidence || []).find(e => e.type === 'change_quantification');
                document.getElementById('spatialNdvi').textContent = (vegEv && vegEv.value) ? ndviFmt(vegEv.value.ndvi_delta) : 'Not available';
                document.getElementById('spatialChangedPx').textContent = (statsEv && statsEv.value) ? pctFmt(statsEv.value.changed_pixel_fraction) : ((vegEv && vegEv.value) ? pctFmt(vegEv.value.affected_fraction) : 'Not available');

            } else {
                // No spatial data available
                document.getElementById('mapPlaceholder').style.display = 'flex';
                const reason = (se && se.reason) ? se.reason : 'Spatial bounds could not be derived from the input.';
                document.getElementById('mapPlaceholderText').textContent = 'Spatial extent unavailable';
                document.getElementById('mapPlaceholderReason').textContent = reason;
                spatialBadge.classList.add('hidden');
                map.setView([0, 0], 2);
                // Clear metadata panel
                ['spatialLabel','spatialLat','spatialLon','spatialCRS','spatialT1','spatialT2','spatialChangedPx','spatialNdvi'].forEach(id => {
                    const el2 = document.getElementById(id);
                    if (el2) el2.textContent = 'Not available';
                });
            }


            // 10 Trace
            const tContent = document.getElementById('uiTrace');
            tContent.innerHTML = '';
            if (data.execution_trace) {
                data.execution_trace.forEach(step => {
                    let dotColor = step.status === "success" ? "bg-emerald-500" : (step.status === "error" ? "bg-red-500" : "bg-slate-400");
                    tContent.innerHTML += `
                        <div class="timeline-step">
                            <div class="timeline-dot ${dotColor}"></div>
                            <div class="flex justify-between items-start">
                                <div><span class="font-bold text-slate-700 block">${step.step}</span><span class="text-[10px] text-slate-400">${step.tool}</span></div>
                                <span class="text-[10px] font-mono text-slate-400">${step.duration_ms}ms</span>
                            </div>
                        </div>
                    `;
                });
            }

            var el = document.getElementById('uiTraceId'); if (el) el.textContent = data.trace_id || "N/A";
            
            // 12 Tools
            const uiV = document.getElementById('uiVersions');
            uiV.innerHTML = '';
            if (data.model_versions) {
                Object.entries(data.model_versions).forEach(([key, val]) => {
                    uiV.innerHTML += `<span class="bg-slate-100 px-2 py-1 rounded border shadow-sm"><b>${key}:</b> ${val}</span>`;
                });
            }
        }
    
