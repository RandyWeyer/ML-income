const net = new brain.NeuralNetwork({
  activation: 'sigmoid', // activation function
  hiddenLayers: [4],
  learningRate: 0.6 // global learning rate, useful when training using streams
});

// var testNet = new brain.recurrent.LSTM();
//
// testNet.train([
//   { input: "13", output: "<=50K" },
//   { input: "11", output: ">50K" }
// ]);
//
// var output = testNet.run("13");
// console.log(output);

// AGE TO INCOME ML
console.log("AGE: ");

var ageNet = new brain.NeuralNetwork();

for (var i = 0; i < 13400; i++) {
  tempInput = incomeDataset[i][0]/100;
  tempOutput = incomeDataset[i][14];
  if (tempOutput == "<=50K") {
    tempOutput = 1;
    ageNet.train([
      { input: {age: tempInput}, output: { lessthan50K: tempOutput } }
    ]);
  } else {
    tempOutput = 1;
    ageNet.train([
      { input: {age: tempInput}, output: { morethan50K: tempOutput }}
    ]);
  }
}

for (var i = 0; i < 100; i++) {
  var ageOutput = ageNet.run({ age: i/100 });
  console.log("Age: " + i);
  console.log(ageOutput);
}

// YEARS OF EDUCATION TO INCOME ML
console.log("EDUCATION: ");

var yearsEducationNet = new brain.NeuralNetwork();

for (var i = 0; i < 13400; i++) {
  tempInput = incomeDataset[i][4]/16;
  tempOutput = incomeDataset[i][14];
  if (tempOutput == "<=50K") {
    tempOutput = 1;
    yearsEducationNet.train([
      { input: {yearsEducation: tempInput}, output: { lessthan50K: tempOutput } }
    ]);
  } else {
    tempOutput = 1;
    yearsEducationNet.train([
      { input: {yearsEducation: tempInput}, output: { morethan50K: tempOutput }}
    ]);
  }
}

for (var i = 1; i < 16; i++) {
  var yearsEducationOutput = yearsEducationNet.run({ yearsEducation: i/16 });
  console.log(yearsEducationOutput);
}

// HOURS WORKED PER WEEK TO INCOME ML
console.log("HOURS WORKED PER WEEK: ");

var weeklyHoursNet = new brain.NeuralNetwork();

for (var i = 0; i < 13400; i++) {
  tempInput = incomeDataset[i][12]/60;
  tempOutput = incomeDataset[i][14];
  if (tempOutput == "<=50K") {
    tempOutput = 1;
    weeklyHoursNet.train([
      { input: {weeklyHours: tempInput}, output: { lessthan50K: tempOutput } }
    ]);
  } else {
    tempOutput = 1;
    weeklyHoursNet.train([
      { input: {weeklyHours: tempInput}, output: { morethan50K: tempOutput }}
    ]);
  }
}

for (var i = 1; i < 60; i++) {
  var weeklyHoursOutput = weeklyHoursNet.run({ weeklyHours: i/60 });
  console.log(weeklyHoursOutput);
}

// GENDER TO INCOME ML
console.log("GENDER: ");
console.log("Male is 0, female is 1");

var genderNet = new brain.NeuralNetwork();

for (var i = 0; i < 13400; i++) {
  tempInput = incomeDataset[i][9];
  if (tempInput == "Male") {
    tempInput = 0;
  } else {
    tempInput = 1;
  }
  tempOutput = incomeDataset[i][14];
  if (tempOutput == "<=50K") {
    tempOutput = 1;
    genderNet.train([
      { input: {gender: tempInput}, output: { lessthan50K: tempOutput } }
    ]);
  } else {
    tempOutput = 1;
    genderNet.train([
      { input: {gender: tempInput}, output: { morethan50K: tempOutput }}
    ]);
  }
}

for (var i = 0; i < 2; i++) {
  var genderOutput = genderNet.run({ gender: i });
  console.log(genderOutput);
}
