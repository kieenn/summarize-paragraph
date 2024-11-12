// summarize handle
$(document).ready(function(){
    const $text = $('.paragraph-box');
    const $message = $(".message");
    const $wordCount = $(".text-info1.word-sentence-count");
    const $resultWordCount = $(".text-info2.word-sentence-count");
    const $resultBox = $(".result-box");

    function validateText() {
        const textValue = $text.val().trim();
        if (textValue.length === 0) {
            $message.text("Please paste your text!!!");
            return false;
        } else {
            $message.empty();
            return true;
        }
    }

function updateCounts() {
    let textValue = $text.val().trim();

    if (!textValue) {
        $wordCount.hide();
        return;
    }
    textValue = textValue.trim();

    // CLEAN TEXT (remove extra newlines, etc.) — Important!
    textValue = textValue.replace(/\s+/g, " ");

    const wordCount = textValue.split(/\s+/).length;

    const sentenceRegex = /[.!?]+(?:["'\)\]\p{Pf}]*)?(?:\s|$)/gu; // Corrected regex
    const sentences = textValue.match(sentenceRegex) || [];
    console.log(sentences)
    const sentenceCount = sentences.length;


    $wordCount.text(`Words: ${wordCount} | Sentences: ${sentenceCount}`).show();
}

    $text.on('input', function() {
        updateCounts();
    });

    $(".summarize-button").click(function() {
        $resultBox.val("")
        $resultWordCount.empty()
        const language_selected = $(".language-select").val().trim();
        const summary_selected = $(".summary-select").val().trim();
        const summary_length = $("#summaryLength").val()
        // alert(summary_length)
        const data = { text: $text.val(), summary_length: summary_length};

        if (validateText()) {
            // Set loading state
            $resultBox.val("Loading...")

            const url = language_selected === "english" ? (summary_selected === "summary1" ? '/summarize' : (summary_selected === "summary2" ? '/summarize2' : '/summarize3')) : '/summarize2';
            

            $.ajax({
                url: url,
                data: JSON.stringify(data),
                method: "POST",
                contentType: 'application/json'
            }).done(function(data) {
                $resultBox.val(data.summary);
                // Update result box word and sentence count
                const resultWordCount = data.summary ? data.summary.split(/\s+/).length : 0;
                const resultSentenceCount = data.summary ? data.summary.split(/[.!?]+(?:["'\)\]\p{Pf}]*)?(?:\s|$)/gu).filter(Boolean).length : 0;
                $resultWordCount.text(`Words: ${resultWordCount} | Sentences: ${resultSentenceCount}`).show();
            }).fail(function(jqXHR) {
                let alertHtml;
                if (jqXHR.status === 400) {
                    alertHtml = `<div class="alert alert-warning alert-dismissible fade show" role="alert" style="position: fixed; top: 10px; right: 10px; z-index: 1050;">
                        Bad Request: Please check your input. </br>
                        Note: Your input must at least 3 sentences.
                        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                    </div>`;
                } else if (jqXHR.status === 500) {
                    $resultWordCount.empty()
                    alertHtml = `<div class="alert alert-danger alert-dismissible fade show" role="alert" style="position: fixed; top: 10px; right: 10px; z-index: 1050;">
                        Error occurred while summarizing. Please try again later. </br>
                        Note: Your input must be at least 3 sentences.
                        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                    </div>`;
                }
                $('body').append(alertHtml);
                setTimeout(() => {
                    $('.alert').alert('close');
                }, 5000);
            })
        }
    });

    // Handle button actions
    $(".btn-outline-primary").click(function() {
        const action = $(this).attr('title').trim();
        const targetTextarea = $(this).closest('.d-flex').siblings('textarea');
        if (action === "Copy") {
            const target = targetTextarea.val();
            navigator.clipboard.writeText(target).then(() => {
                // Create a Bootstrap 5 alert
                const alertHtml = `<div class="alert alert-success alert-dismissible fade show" role="alert" style="position: fixed; top: 10px; right: 10px; z-index: 1050;">
                    Text copied to clipboard!
                    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                </div>`;
                // Append the alert to the body or a specific container
                $('body').append(alertHtml);
                // Automatically remove the alert after 3 seconds
                setTimeout(() => {
                    $('.alert').alert('close');
                }, 3000);
            });
        } else if (action === "Paste") {
            navigator.clipboard.readText().then(text => {
                $text.val(text);
                updateCounts();
            });
        } else if (action === "Clear") {
            targetTextarea.val('');
            updateCounts();
        }
    });
});