document.addEventListener('DOMContentLoaded', () => {
    const queryForm = document.getElementById('query-form');
    const queryText = document.getElementById('query-text');
    const queryImage = document.getElementById('query-image');
    const imagePreview = document.getElementById('image-preview');
    const submitButton = document.getElementById('submit-button');

    const resultsArea = document.getElementById('results-area');
    const statusMessage = document.getElementById('status-message');
    const generationSection = document.getElementById('generation-section');
    const generatedResponseDiv = document.getElementById('generated-response');
    const retrievalSection = document.getElementById('retrieval-section');
    const retrievedDocsDiv = document.getElementById('retrieved-docs');
    const systemStatusDiv = document.getElementById('system-status');

    // 图片弹窗相关
    const imageModal = document.getElementById('image-modal');
    const modalImg = document.getElementById('modal-img');
    // 页面初始时确保弹窗隐藏且图片src为空
    if (imageModal) {
        imageModal.classList.add('hidden');
        imageModal.style.display = 'none';
    }
    if (modalImg) modalImg.src = '';

    // 打开弹窗时显示，关闭时隐藏
    function showImageModal(src) {
        if (imageModal && modalImg) {
            modalImg.src = src;
            imageModal.classList.remove('hidden');
            imageModal.style.display = 'flex';
        }
    }
    function hideImageModal() {
        if (imageModal && modalImg) {
            imageModal.classList.add('hidden');
            imageModal.style.display = 'none';
            modalImg.src = '';
        }
    }

    // Function to update status message display
    function updateStatus(message, type = 'processing') {
        if (!statusMessage) return;
        statusMessage.textContent = message;
        statusMessage.className = `status-box status-${type}`; // Add type class
        statusMessage.classList.remove('hidden');
        console.log(`Status Update [${type}]: ${message}`);
    }

    // Function to update system status display
    function updateSystemStatus(message, type = 'pending') {
         if (!systemStatusDiv) return;
         systemStatusDiv.textContent = `System Status: ${message}`;
         systemStatusDiv.className = `status-box status-${type}`;
         console.log(`System Status Update [${type}]: ${message}`);
    }


    // Function to clear results
    function clearResults() {
        if (generatedResponseDiv) generatedResponseDiv.innerHTML = '';
        if (retrievedDocsDiv) retrievedDocsDiv.innerHTML = '';
        if (generationSection) generationSection.classList.add('hidden');
        if (retrievalSection) retrievalSection.classList.add('hidden');
        if (statusMessage) statusMessage.classList.add('hidden');
        if (statusMessage) statusMessage.textContent = '';
        if (statusMessage) statusMessage.className = 'status-box'; // Reset class
    }

    // Image preview handler
    queryImage.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                imagePreview.classList.remove('hidden');
            }
            reader.readAsDataURL(file);
        } else {
            imagePreview.src = '#';
            imagePreview.classList.add('hidden');
        }
    });

    // Check system status on load
    async function checkSystemStatus() {
        updateSystemStatus("Checking...", "pending");
        try {
            const response = await fetch('/status');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            console.log("System Status Response:", data);

            if (data.status === "success") {
                updateSystemStatus("Ready", "success");
                submitButton.disabled = false;
            } else if (data.status === "partial") {
                 const details = [];
                 if(data.indexer !== 'success') details.push('Indexer issue');
                 if(data.retriever !== 'success') details.push('Retriever issue');
                 if(data.generator !== 'success') details.push('Generator issue');
                 updateSystemStatus(`Partial (${details.join(', ')})`, "warning");
                 // Decide if partial is usable - maybe disable submit if retriever is down?
                 submitButton.disabled = (data.retriever === 'failed' || data.retriever === 'skipped (indexer failed)');
                 if(submitButton.disabled) {
                     updateStatus("System partially initialized. Querying disabled due to critical component failure (Retriever).", "error");
                 } else {
                      updateStatus("System partially initialized. Some features (like generation) might be unavailable.", "warning");
                 }
            }
             else { // failed or other status
                updateSystemStatus(`Failed (${data.error_message || 'Unknown reason'})`, "error");
                updateStatus(`System initialization failed: ${data.error_message || 'Unknown reason'}. Querying disabled.`, "error");
                submitButton.disabled = true;
            }
        } catch (error) {
            console.error("Error checking system status:", error);
            updateSystemStatus(`Error checking status (${error.message})`, "error");
            updateStatus(`Could not verify system status: ${error.message}. Querying disabled.`, "error");
            submitButton.disabled = true;
        }
    }

    // Function to show loading in answer/retrieval columns
    function showLoadingPlaceholders() {
        if (generationSection) {
            generationSection.classList.remove('hidden');
            generatedResponseDiv.innerHTML = '<div class="loading-spinner"></div><span style="margin-left:12px;">答案生成中...</span>';
        }
        if (retrievalSection) {
            retrievalSection.classList.remove('hidden');
            retrievedDocsDiv.innerHTML = '<div class="loading-spinner"></div><span style="margin-left:12px;">知识检索中...</span>';
        }
    }

    // Form submission handler
    queryForm.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent default form submission
        clearResults();
        showLoadingPlaceholders(); // 显示加载动画

        const text = queryText.value.trim();
        const imageFile = queryImage.files[0];

        if (!text && !imageFile) {
            updateStatus("Please enter a text query or upload an image.", 'error');
            return;
        }

        updateStatus("Processing your query...", 'processing');
        submitButton.disabled = true;
        submitButton.innerHTML = '<span class="loading-spinner" style="vertical-align:middle;margin-right:8px;"></span>提交中...';

        const formData = new FormData();
        if (text) {
            formData.append('query_text', text);
        }
        if (imageFile) {
            formData.append('query_image', imageFile);
        }

        try {
            const response = await fetch('/query', {
                method: 'POST',
                body: formData,
                // 'Content-Type' header is automatically set by browser for FormData
            });

            if (!response.ok) {
                // Try to get error detail from response body
                let errorDetail = `HTTP error! status: ${response.status}`;
                try {
                     const errorData = await response.json();
                     errorDetail = errorData.detail || JSON.stringify(errorData);
                } catch (jsonError) {
                    // If response is not JSON, use status text
                    errorDetail = response.statusText || errorDetail;
                }
                throw new Error(errorDetail);
            }

            const data = await response.json();
            console.log("Received data:", data);

            // Display results
            displayResults(data);
            updateStatus("Query processed successfully!", 'success');

        } catch (error) {
            console.error("Error submitting query:", error);
            updateStatus(`Error: ${error.message}`, 'error');
            if (resultsArea) resultsArea.classList.remove('hidden'); // Ensure results area is visible to show error
            if (typeof generationSection !== 'undefined' && generationSection && generationSection.classList && typeof generationSection.classList.add === 'function') {
                generationSection.classList.add('hidden');
            }
            if (typeof retrievalSection !== 'undefined' && retrievalSection && retrievalSection.classList && typeof retrievalSection.classList.add === 'function') {
                retrievalSection.classList.add('hidden');
            }
        } finally {
            submitButton.disabled = false; // Re-enable button
            submitButton.innerHTML = '提交问题';
        }
    });

    // Function to display results in the HTML
    function displayResults(data) {
        // Display Generated Response
        if (data.generated_response) {
            generatedResponseDiv.textContent = data.generated_response;
            generationSection.classList.remove('hidden');
        } else {
            generatedResponseDiv.textContent = 'No response generated.';
            generationSection.classList.remove('hidden');
        }

        // Display Retrieved Documents
        if (data.retrieved_docs && data.retrieved_docs.length > 0) {
            retrievedDocsDiv.innerHTML = '';
            data.retrieved_docs.forEach(doc => {
                const docCard = document.createElement('div');
                docCard.classList.add('doc-card');
                docCard.style.marginBottom = '28px';
                docCard.style.background = 'linear-gradient(135deg, #f8fafc 60%, #e0e7ff 100%)';
                docCard.style.borderRadius = '14px';
                docCard.style.boxShadow = '0 4px 16px rgba(60,60,120,0.10)';
                docCard.style.padding = '22px 18px 18px 18px';
                docCard.style.transition = 'box-shadow 0.2s';
                docCard.onmouseover = () => docCard.style.boxShadow = '0 8px 32px rgba(60,60,120,0.18)';
                docCard.onmouseout = () => docCard.style.boxShadow = '0 4px 16px rgba(60,60,120,0.10)';

                const score = typeof doc.score === 'number' ? doc.score.toFixed(4) : doc.score || 'N/A';
                const textContent = doc.text ? doc.text : 'No text content available.';
                let cardInnerHtml = '';
                if (doc.image_path) {
                    // 有图片，左右结构
                    const imgFileName = doc.image_path.split(/[\\\/]/).pop();
                    const imgId = `img_${Math.random().toString(36).substr(2, 9)}`;
                    cardInnerHtml = `
                        <div style="display:flex;flex-direction:row;gap:24px;align-items:flex-start;">
                            <div style="flex:1;min-width:0;">
                                <p style="margin-bottom:6px;"><strong class="doc-id" style="color:#2563eb;">ID:</strong> ${doc.id || 'N/A'}</p>
                                <p style="margin-bottom:6px;"><strong class="doc-score" style="color:#059669;">Score:</strong> ${score}</p>
                                <div class="doc-text" style="max-height:none;overflow:visible;white-space:pre-wrap;font-size:1.08em;color:#22223b;background:#f1f5f9;border-radius:8px;padding:12px 14px;margin:10px 0 0 0;line-height:1.7;">
                                    <strong style="color:#6366f1;">Text:</strong> <span style='word-break:break-all;'>${textContent}</span>
                                </div>
                            </div>
                            <div style="flex:0 0 220px;max-width:220px;">
                                <div style=\"margin-bottom:10px;\"><strong style=\"color:#6366f1;\">图片预览:</strong><br>
                                    <a href="/images/${imgFileName}" target="_blank" title="点击新标签页全屏预览">
                                        <img id="${imgId}" src="/images/${imgFileName}" alt="${imgFileName}" style="max-width:100%;height:auto;border-radius:10px;box-shadow:0 2px 8px rgba(60,60,120,0.13);margin-top:8px;cursor:pointer;"
                                        onerror="this.style.display='none';var errMsg=document.createElement('div');errMsg.style.color='red';errMsg.textContent='图片加载失败: ' + this.src;this.parentNode.appendChild(errMsg);console.error('图片加载失败:', this.src);">
                                    </a>
                                </div>
                                <div style='word-break:break-all;font-size:0.98em;color:#888;margin-bottom:8px;'><strong>图片路径:</strong> ${doc.image_path}</div>
                            </div>
                        </div>
                    `;
                } else {
                    // 无图片，文字内容铺满卡片
                    cardInnerHtml = `
                        <div style="display:flex;flex-direction:column;align-items:stretch;">
                            <p style="margin-bottom:6px;"><strong class="doc-id" style="color:#2563eb;">ID:</strong> ${doc.id || 'N/A'}</p>
                            <p style="margin-bottom:6px;"><strong class="doc-score" style="color:#059669;">Score:</strong> ${score}</p>
                            <div class="doc-text" style="max-height:none;overflow:visible;white-space:pre-wrap;font-size:1.08em;color:#22223b;background:#f1f5f9;border-radius:8px;padding:12px 14px;margin:10px 0 0 0;line-height:1.7;">
                                <strong style="color:#6366f1;">Text:</strong> <span style='word-break:break-all;'>${textContent}</span>
                            </div>
                        </div>
                    `;
                }
                docCard.innerHTML = cardInnerHtml;
                retrievedDocsDiv.appendChild(docCard);
            });
            retrievalSection.classList.remove('hidden');
        } else {
            retrievedDocsDiv.innerHTML = '<p>No relevant documents were retrieved.</p>';
            retrievalSection.classList.remove('hidden');
        }

        // Display top-level error if present
         if(data.error) {
             updateStatus(`Processing finished with errors: ${data.error}`, 'error');
         }
    }

    // 事件代理：点击检索区图片弹窗大图
    retrievedDocsDiv.addEventListener('click', function(e) {
        const target = e.target;
        if (target.tagName === 'IMG' && target.closest('a') && target.closest('a').getAttribute('href').startsWith('/images/')) {
            e.preventDefault();
            showImageModal(target.src);
        }
    });
    // 点击弹窗遮罩关闭
    if (imageModal) {
        imageModal.addEventListener('click', function(e) {
            if (e.target === imageModal) {
                hideImageModal();
            }
        });
    }
    // ESC键关闭弹窗
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && imageModal && imageModal.style.display !== 'none') {
            hideImageModal();
        }
    });

    // Initial check
    checkSystemStatus();
});