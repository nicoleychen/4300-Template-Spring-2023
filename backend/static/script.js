//init
function init() {
  console.log(document.readyState);
}
window.onload = init();
console.log("Window onload is", window.onload);

// creates API request every time a gender pref is selected
// if (document.querySelector('input[name="gender"]')) {
//   document.querySelectorAll('input[name="gender"]').forEach((elem) => {
//     elem.addEventListener("change", function(event) {
//     var gender = event.target.value;
//     fetch("/gender_pref?" + new URLSearchParams({ gender: gender }).toString())
//     .then(function (response){
//       return response.json();
//     }).then(function (data) {
//       console.log(data);
//     }).catch(function (err){
//       console.warn("Something went wrong.", err);
//     });
//      });
//   });
// }

const button = document.getElementById('search-button')
const perfSearchBox = document.querySelector(".input-box")
const perfAutoBox = document.querySelector("#perf-auto-box")
const perfInputBox = document.querySelector("#search-text")
console.log(button)
button.addEventListener('click', returnResults)

function returnResults() {
  // const removeText = document.getElementsByClassName("no-result");
  // removeText[0].parentNode.removeChild(removeText[0]);

  const removeBoxes = document.getElementsByClassName('result-card')
  while (removeBoxes.length > 0) {
    removeBoxes[0].parentNode.removeChild(removeBoxes[0])
  }

  let query = document.getElementById('search-text').value
  let male_pref = document.getElementById('male-button').checked
  let female_pref = document.getElementById('female-button').checked
  let no_pref = document.getElementById('no-pref-button').checked

  let gender_filter = ''
  if (male_pref) {
    gender_filter = 'for women'
  } else if (female_pref) {
    gender_filter = 'for men'
  } else {
    gender_filter = ''
  }

  fetch(
    '/self?' +
      new URLSearchParams({
        name: query,
      }).toString()
  ).then(function (response) {
    return response.json()
  })
  .then(function (data) {
    const res = data
    let tempDiv = document.createElement('div')
          tempDiv.innerHTML = resultTemplate(
            res.img,
            res.name,
            res.brand,
            res.rating,
            res.gender,
            res.topnote,
            res.middlenote,
            res.bottomnote,
            res.desc
          )
    document.getElementById('query-box').appendChild(tempDiv)
  })

  document.getElementById('result-heading-text').textContent =
    'Results similar to "' + query + '":'

  fetch(
    '/similar?' +
      new URLSearchParams({
        name: query,
        gender_pref: gender_filter
      }).toString()
  )
    .then(function (response) {
      // The API call was successful!
      return response.json()
    })
    .then(function (data) {
      // This is the JSON from our response
      console.log(data)
      const results = data
      if (Object.values(results).length === 0) {
        console.log('NOT FOUND IN DATASET')
        let tempDiv = document.createElement('div')
        tempDiv.innerHTML = noResultTemplate()
        document.getElementById('result-box').appendChild(tempDiv)
      } else {
        results.forEach(res => {
          let tempDiv = document.createElement('div')
          tempDiv.innerHTML = resultTemplate(
            res.img,
            res.name,
            res.brand,
            res.rating,
            res.gender,
            res.topnote,
            res.middlenote,
            res.bottomnote,
            res.desc
          )
          document.getElementById('result-box').appendChild(tempDiv)
        })
      }
    })
    .catch(function (err) {
      // There was an error
      console.warn('Something went wrong.', err)
    })
}

function resultTemplate(
  img,
  name,
  brand,
  rating,
  gender,
  topnote,
  middlenote,
  bottomnote,
  desc
) {
  return `<div class='result-card'>
            <img src=${img} class='fragrance-img'>
            <h3 class='fragrance-name'>${name} by ${brand}</h3>
            <p class = 'fragrance-detail'> Rating: ${rating}</p> 
            <p class = 'fragrance-detail'> Gender: ${gender}</p> 
            <p class = 'fragrance-detail'> Top notes: ${topnote} </p>
            <p class = 'fragrance-detail'> Middle notes: ${middlenote} </p>
            <p class = 'fragrance-detail'> Base notes: ${bottomnote}</p>
            <p class='fragrance-detail'>Description: ${desc}</p>
        </div>`
}

function noResultTemplate() {
  return `<div class = 'result-card'> No results found.</div>`
}

function setPerfClickable(list){
  for(let i=0;i<list.length;i++){
    list[i].setAttribute("onclick","selectPerf(this)")
  }
}

function selectPerf(element){
  let selectUserData = element.textContent;
  perfInputBox.value=selectUserData;
  perfSearchBox.classList.remove("active")
}

function loadPerfSuggestion(){
  perfInputBox.onkeyup = (e)=>{
    let userData = e.target.value
    let emptyArray = []
    let allList = []
    if(userData!=""){
      fetch(
          "/suggestion/perf?" +
          new URLSearchParams({
            name: userData,
          }).toString()
        ).then((response) => response.json())
        .then((data) =>
          emptyArray = data,
        ).then(()=>{
          emptyArray = emptyArray.map((i)=>{
          return i = "<li>"+i+"</li>"
          }),
          (
            perfSearchBox.classList.add("active"),
            perfAutoBox.innerHTML = emptyArray.join(''),
            allList = perfAutoBox.querySelectorAll("li"),
            setPerfClickable(allList)
          )
        }
      );
    }else{
      perfSearchBox.classList.remove("active")
    }
  }
}