// summarize handle
$(document).ready(function(){
    const $text = $('.paragraph-box');
    const $message = $(".message");
    const $wordCount = $(".text-info1.word-sentence-count");
    const $resultWordCount = $(".text-info2.word-sentence-count");

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
        const textValue = $text.val().trim();
        const wordCount = textValue ? textValue.split(/\s+/).length : 0;
        const sentenceCount = textValue ? textValue.split(/[.!?]+/).filter(Boolean).length : 0;
        if (wordCount === 0 && sentenceCount === 0) {
            $wordCount.hide();
        } else {
             $message.empty();
            $wordCount.text(`Words: ${wordCount} | Sentences: ${sentenceCount}`).show();
        }
    }

    $text.on('input', function() {
        updateCounts();
    });

    $(".summarize-button").click(function() {
        const language_selected = $(".language-select").val().trim();
        const summary_selected = $(".summary-select").val().trim();
        const data = { text: $text.val() };

        if (validateText()) {
            const url = language_selected === "english" ? (summary_selected === "summary1" ? '/summarize' : (summary_selected === "summary2" ? '/summarize2' : '/summarize3')) : '/summarize2';
            $.ajax({
                url: url,
                data: JSON.stringify(data),
                method: "POST",
                contentType: 'application/json'
            }).done(function(data) {
                $(".result-box").empty()
                $(".result-box").val(data.summary);
                // Update result box word and sentence count
                const resultWordCount = data.summary ? data.summary.split(/\s+/).length : 0;
                const resultSentenceCount = data.summary ? data.summary.split(/[.!?]+/).filter(Boolean).length : 0;
                $resultWordCount.text(`Words: ${resultWordCount} | Sentences: ${resultSentenceCount}`);
            });
        }
    });

    // Handle button actions
    $(".btn-outline-primary").click(function() {
        const action = $(this).attr('title').trim();
        if (action === "Copy") {
            const target = $(this).closest('.d-flex').siblings('textarea').val();
            navigator.clipboard.writeText(target).then(() => {
                $message.text("Text copied to clipboard!").fadeIn().delay(3000).fadeOut();
            });
        } else if (action === "Paste") {
            navigator.clipboard.readText().then(text => {
                $text.val(text);
                updateCounts();
            });
        }
    });
});