// summarize handle
$(document).ready(function(){
  $(".summarize-button").click(function(){
      const text = $('.paragraph-box').val();
      $('.result-box').prop('value',text);
  });
});