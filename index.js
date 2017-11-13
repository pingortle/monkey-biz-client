const _ = require('lodash')
const { BrainBuilder, Trainer, Activators } = require('brain')

const brain = BrainBuilder.new(new Activators.Sigmoid)
    .inputs(2)
    .layer(2)
    .layer(2)
    .outputs(1)
    .build()

const xorData = [
  [[0, 0], [0]],
  [[1, 0], [1]],
  [[0, 1], [1]],
  [[1, 1], [0]],
]

const trainingSet = xorData

const trainer = new Trainer(brain, { learningRate: .2 })

const beforeError = trainingSet.reduce((error, [input, output]) => {
    const reaction = brain.react(input)

    let errors = _.zipWith(reaction, output, _.subtract).map(Math.abs)
    if (_.isNumber(error)) errors = errors.concat(error)

    return _.mean(errors)
  })

_.times(10000, () => {
    trainingSet.forEach(([input, output]) => {
      trainer.train(input, output)
    })
  })

const afterError = trainingSet.reduce((error, [input, output]) => {
    const reaction = brain.react(input)

    let errors = _.zipWith(reaction, output, _.subtract).map(Math.abs)
    if (_.isNumber(error)) errors = errors.concat(error)

    return _.mean(errors)
  })

console.log(`error before: ${beforeError}, error after: ${afterError}`)
