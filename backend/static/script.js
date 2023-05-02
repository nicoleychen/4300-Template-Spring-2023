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
const perfSearchBox = document.querySelector("#perf-search-box")
const perfAutoBox = document.querySelector("#perf-auto-box")
const perfInputBox = document.querySelector("#search-text")
console.log(button)
button.addEventListener('click', returnResults_search)
let votes = new Map();
let like_list = [];
let dislike_list = [];

function returnResults_search(){
  console.log("returnResults_search is called!")
  votes = new Map();
  like_list = [];
  dislike_list = [];
  document.getElementById("like-list").innerHTML = '';
  document.getElementById("dislike-list").innerHTML = '';

  returnResults();
}

function returnResults() {
  console.log("returnResults is called!")

  // const removeText = document.getElementsByClassName("no-result");
  // removeText[0].parentNode.removeChild(removeText[0]);

  const removeBoxes = document.getElementsByClassName('result-card')
  while (removeBoxes.length > 0) {
    removeBoxes[0].parentNode.removeChild(removeBoxes[0])
  }
  const removeText = document.getElementsByClassName('no-result')
  while (removeText.length > 0) {
    removeText[0].parentNode.removeChild(removeText[0])
  }
  const removeQueryBox = document.getElementsByClassName('query-box')
  while (removeQueryBox.length > 0) {
    removeQueryBox[0].parentNode.removeChild(removeQueryBox[0])
  }

  let query = document.getElementById('search-text').value
  let male_pref = document.getElementById('male-button').checked
  let female_pref = document.getElementById('female-button').checked
  let no_pref = document.getElementById('no-pref-button').checked
  let min_rating_input = document.getElementById('rating').value

  console.log(min_rating_input)
  console.log(like_list)
  console.log(dislike_list)

  let gender_filter = ''
  if (male_pref) {
    gender_filter = 'for women'
  } else if (female_pref) {
    gender_filter = 'for men'
  } else if (no_pref) {
    gender_filter = ''
  } else {
    gender_filter = ''
  }

  document.getElementById('query-heading-text').textContent = '"' + query + '":'

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
    if (Object.values(res).length === 0) {
      console.log('NOT FOUND IN DATASET')
      let tempDiv = document.createElement('div')
      tempDiv.innerHTML = noQueryResultTemplate()
      document.getElementById('query-box').appendChild(tempDiv)
    } else {
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
    }
  })

  document.getElementById('result-heading-text').textContent =
    'Results similar to "' + query + '":'

  fetch(
    '/similar?' +
      new URLSearchParams({
        name: query,
        gender_pref: gender_filter,
        min_rating: min_rating_input,
        rel_list: like_list,
        irrel_list: dislike_list
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
  like_disabled = "";
  dislike_disabled = "";
  happy_button = "happy_empty.png";
  sad_button = "sad_empty.png";
  if (votes.has(name)) {
    if (votes.get(name) === 1) {
      like_disabled = "disabled: disabled";
      happy_button = "happy_filled.png";
    } else if (votes.get(name) === -1) {
      dislike_disabled = "disabled: disabled";
      sad_button = "sad_filled.png";
    }
  }
  // name = JSON.stringify(name).replace(/&/, "&amp;").replace(/'/g, "&quot;")
  return `<div class='result-card'>
        <img src=${img} class='fragrance-img'>
        <h3 class='fragrance-name'>${name} by ${brand}</h3>
        <p class = 'fragrance-detail'> Rating: ${rating}</p> 
        <p class = 'fragrance-detail'> Gender: ${gender}</p> 
        <p class = 'fragrance-detail'> Top notes: ${topnote} </p>
        <p class = 'fragrance-detail'> Middle notes: ${middlenote} </p>
        <p class = 'fragrance-detail'> Base notes: ${bottomnote}</p>
        <p class='fragrance-detail'>Description: ${desc}</p>

        <div class="vote-button-group">
          <button class="vote_button" type="submit" ${like_disabled} onclick="updateRelevance(\'${name}\', 1)" id="like-button-${name}">
            <img src="/static/images/${happy_button}" id="like-${name}" alt="Like"/>
          </button>
          <button class="vote_button" type="submit" ${dislike_disabled} onclick="updateRelevance(\'${name}\', -1)" id="dislike-button-${name}">
            <img src="/static/images/${sad_button}" id="dislike-${name}" alt="Dislike"/>
          </button>
        </div>
  </div>`
}

function noQueryResultTemplate() {
  return `<div class = 'no-result'> There is no perfume by that name in our database. Try double checking your spelling.</div>`
}

function noResultTemplate() {
  return `<div class = 'no-result'> No similar perfumes found. Try modifying your query.</div>`
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

function updateRelevance(name, update) {
  votes.set(name, update);
  if (update === 1) {
    document.getElementById("like-button-" + name).disabled = true;
    document.getElementById("dislike-button-" + name).disabled = false;
    document.getElementById("like-" + name).src =
      "/static/images/happy_filled.png";
    document.getElementById("dislike-" + name).src =
      "/static/images/sad_empty.png";
    // document.getElementById("dislike-button-" + name).img = (
    //   <img src="/static/images/up_clicked.png" alt="Thumb Up" />
    // );
  } else if (update === -1) {
    document.getElementById("dislike-button-" + name).disabled = true;
    document.getElementById("like-button-" + name).disabled = false;
    document.getElementById("like-" + name).src = "/static/images/happy_empty.png";
    document.getElementById("dislike-" + name).src =
      "/static/images/sad_filled.png";
  }
  update_like_dislike_list();
}

function update_like_dislike_list() {
  like_list_string = "";
  dislike_list_string = "";
  like_list = []
  dislike_list = []
  for (let [perf, vote] of votes) {
    if (vote === 1) {
      like_list_string += "<p>" + perf + "</p>";
      if (!like_list.includes(perf)) {
        // ✅ only runs if value not in array
        like_list.push(perf);
      }
    } else if (vote === -1) {
      dislike_list_string += "<p>" + perf + "</p>";
      if (!dislike_list.includes(perf)) {
        // ✅ only runs if value not in array
        dislike_list.push(perf);
      }
    }
  }
  document.getElementById("like-list").innerHTML = like_list_string;
  document.getElementById("dislike-list").innerHTML = dislike_list_string;

  returnResults()
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