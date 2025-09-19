// 添加事件监听器来处理分页点击
$(document).on('click', '#pagination .page-link', function(e) {
    e.preventDefault();
    const page = $(this).data('page');
    changePage(page);
});


function _getDailyFoodGroups(mealPlan,detailsData){
  // 使用数组维护 daily_food_id 的顺序
  let dailyFoodOrder = [];
  // 使用 Map 维护 daily_food_id 到 items 的映射
      let dailyFoodGroups = new Map();
      // 定义 level 的排序顺序
      const levelOrder = { "gold": 0, "silver": 1, "bronze": 2 };
      if (Object.keys(mealPlan).length > 0) {
        mealPlan.sort((a, b) => {
          return levelOrder[a.level] - levelOrder[b.level];
        });
        for (let item of mealPlan) {
          if (!dailyFoodGroups.has(item.daily_food_id)) {
            dailyFoodGroups.set(item.daily_food_id, []);
            dailyFoodOrder.push(item.daily_food_id);
          }
          const foundDetail = detailsData.filter(detail => detail.daily_food_id === item.daily_food_id);
          dailyFoodGroups.get(item.daily_food_id).push(foundDetail);
        }
      }
      return { dailyFoodGroups, dailyFoodOrder };
  }

function render(targetRange,targetData,mealPlan,detailsData){

      let currentPage = 0;
      window.changePage = function(page) {
          currentPage = page;
          updateTable(targetRange,targetData,mealPlan,detailsData,currentPage);
          event.preventDefault();
          $('html, body').animate({
            scrollTop: $("#mealPlanTable").offset()
          }, 500);
      };
      updateTable(targetRange,targetData,mealPlan,detailsData,currentPage);
  }

function updateRadar(targetRange,targetData,foundMeal) {
    const radarContainer = document.querySelector('.radar-container');
    const legendExplanation = document.querySelector('.legend-explanation');
    if (Object.keys(targetRange).length > 0) {
        // 如果有数据，显示雷达图和解释
        radarContainer.style.display = 'block';
        legendExplanation.style.display = 'block';
        const radarChart = echarts.init(document.getElementById('radarChart'));
        const radarRangeData = Object.entries(targetRange).map(([key, [min, max]]) => {
          return {
            name: key.replace('_', ' '),
            min: parseFloat(min).toFixed(2),
            max: parseFloat(max).toFixed(2)
          };
        });

        const radarNutriData = Object.entries(foundMeal[0]).map(([key, value]) => {
          return {
            name: key.replace('_', ' '),
            value: parseFloat(value).toFixed(2)
          };
        });

        const radarTargetData = Object.entries(targetData).map(([key, value]) => {
          return {
            name: key.replace('_', ' '),
            value: parseFloat(value).toFixed(2)
          };
        });

        const indicators = radarRangeData.map(item => ({ name: item.name }));
        const maxValues = radarRangeData.map(item => item.max);
        const minValues = radarRangeData.map(item => item.min);
        const targetValues = radarTargetData.map(item => item.value);
        const nutriValues = indicators.map(indicator => {
          const matchedItem = radarNutriData.find(item => item.name === indicator.name);
          return matchedItem ? parseFloat(matchedItem.value) : null;
        });

        const radarChartOption = {
          tooltip: {
            trigger: 'item'
          },
          legend: {
            data: ['Target Intake', 'Max Intake', 'Min Intake', 'Meal plan'],
            top: 'bottom'
          },
          radar: {
            indicator: indicators,
            radius: '65%',
            splitNumber: 5,
            axisName: {
              fontSize: 12
            },
            splitArea: {
              show: true,
              areaStyle: {
                color: ['rgba(255, 255, 255, 0.8)', 'rgba(255, 255, 255, 0.6)']
              }
            }
          },
          series: [
            {
              name: 'Target Intake',
              type: 'radar',
              data: [targetValues],
              lineStyle: {
                color: 'rgba(75, 192, 192, 1)',
                type: 'dashed'
              },
              itemStyle: {
                color: 'rgba(75, 192, 192, 1)'
              }
            },
            {
              name: 'Max Intake',
              type: 'radar',
              data: [maxValues],
              lineStyle: {
                color: 'rgba(255, 99, 132, 1)',
                type: 'dashed'
              },
              itemStyle: {
                color: 'rgba(255, 99, 132, 1)'
              }
            },
            {
              name: 'Min Intake',
              type: 'radar',
              data: [minValues],
              lineStyle: {
                color: 'rgba(54, 162, 235, 1)',
                type: 'dashed'
              },
              itemStyle: {
                color: 'rgba(54, 162, 235, 1)'
              }
            },
            {
              name: 'Meal plan',
              type: 'radar',
              data: [nutriValues],
              lineStyle: {
                color: 'rgba(153, 102, 255, 1)',
                width: 2
              },
              itemStyle: {
                color: 'rgba(153, 102, 255, 1)',
              }
            }
          ]
        };
        radarChart.setOption(radarChartOption);
     } else {
        radarContainer.style.display = 'none';
        legendExplanation.style.display = 'none';
      }
  }

function updateTable(targetRange,targetData,mealPlan,detailsData,currentPage) {
      const tableBody = $('#mealPlanTableBody');
      tableBody.empty();
      const dailyFood= _getDailyFoodGroups(mealPlan,detailsData)
      const dailyFoodOrder=dailyFood.dailyFoodOrder;
      const dailyFoodGroups=dailyFood.dailyFoodGroups;
      if (dailyFoodOrder.length > 0) {
        const detailsContainer = document.getElementById('details-container');
        detailsContainer.style.display = 'block'; // 显示容器
        const dailyFoodId = dailyFoodOrder[currentPage];
        const foundMeal = mealPlan.filter(meal => meal.daily_food_id === dailyFoodId);
        const level = foundMeal[0].level;
        $('#tableTitle').text(level + ' - Daily Food ID: ' + dailyFoodId);
        const items = dailyFoodGroups.get(dailyFoodId);
        updateRadar(targetRange,targetData,foundMeal);
        for (const item of items[0]) {
          tableBody.append(`
            <tr>
              <td>${item.food_id}</td>
              <td>${item.food_desc}</td>
              <td>${item.food_desc_long}</td>
              <td>${item.grams}</td>
              <td>${item.eating_type}</td>
            </tr>
          `);
        }
        const totalPages=dailyFoodGroups.size;
        updatePagination(currentPage,totalPages);
        document.getElementById('exportCsvBtn').addEventListener('click',()=>{exportToCSV(detailsData)} );
      }
  }

function updatePagination(currentPage, totalPages) {
    const pagination = $('#pagination');
    pagination.empty();

    // 创建分页项的辅助函数
    function createPageItem(text, pageNum, isDisabled = false, isActive = false) {
        const disabledClass = isDisabled ? 'disabled' : '';
        const activeClass = isActive ? 'active' : '';
        return `
            <li class="page-item ${disabledClass} ${activeClass}">
                <a class="page-link" href="#" data-page="${pageNum}">${text}</a>
            </li>
        `;
    }

    // 添加第一页按钮
    pagination.append(createPageItem('First', 0, currentPage === 0));

    // 添加上一页按钮
    pagination.append(createPageItem('Previous', currentPage - 1, currentPage === 0));

    // 添加页码按钮
    const maxVisiblePages = 3;
    let startPage = Math.max(0, currentPage - Math.floor(maxVisiblePages / 2));
    let endPage = Math.min(totalPages - 1, startPage + maxVisiblePages - 1);

    if (endPage - startPage + 1 < maxVisiblePages) {
        startPage = Math.max(0, endPage - maxVisiblePages + 1);
    }

    for (let i = startPage; i <= endPage; i++) {
        pagination.append(createPageItem(i + 1, i, false, i === currentPage));
    }

    // 添加下一页按钮
    pagination.append(createPageItem('Next', currentPage + 1, currentPage === totalPages - 1));

    // 添加最后一页按钮
    pagination.append(createPageItem('Last', totalPages - 1, currentPage === totalPages - 1));
}


function exportToCSV(detailsData) {
    const dailyFoodGroups = detailsData.reduce((acc, item) => {
      if (!acc[item.daily_food_id]) {
        acc[item.daily_food_id] = [];
      }
      acc[item.daily_food_id].push(item);
      return acc;
    }, {});

    const dailyFoodIds = Object.keys(dailyFoodGroups);

    fetch('/export-csv', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(dailyFoodGroups),
    })
    .then(response => response.blob())
    .then(blob => {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = 'meal_plan.csv';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
    });
  }