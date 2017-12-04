const zlib = require('zlib');
const JSONStream = require('JSONStream');
const fs = require('fs');
const csv = require('fast-csv');

const fileName = './data/matches_small.csv';
const write = fs.createReadStream(fileName);
var count = 0;

// var myObj = { "name":"John", "age":31, "city":"New York" };
// var myJSON = JSON.stringify(myObj);
// console.log(myJSON)

csv
	.fromStream(write, {headers: ["match_id",,,,,,,,,,,,,,,,,,,,,"chat",,,,,,], ignoreEmpty: true, objectMode:false})
	.on('data', (data) => {
		console.log(data)
		if (data.chat !== '' && count < 2){
			var chatString = data.chat.substr(1, data.chat.length-2);
			chatString = "\"{\"chat\":[" + chatString + "]}\""
			console.log(chatString);
			count+=1;
		}
	})
	.on('error', (err, data) => {
		console.log('Some error:');
		console.log(err);
		console.log(data);
	});
