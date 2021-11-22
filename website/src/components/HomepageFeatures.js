import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'Find Quality Model at Your Fingertips',
    Svg: require('../../static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        FLAML finds accurate ML models with small computational resources
        for common ML tasks like classification and regression.
        It frees users from selecting learners and hyperparameters.
        {/* It is fast and economical. */}
      </>
    ),
  },
  {
    title: 'Easy to Customize or Extend',
    Svg: require('../../static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        FLAML is designed easy to extend, such as adding custom learners or metrics.
        The customizaation level can range from minimal
(training data and task type as only input) to full (tuning a user-defined function).
      </>
    ),
  },
  {
    title: 'Auto Tuning: Power Up, Cost Down',
    Svg: require('../../static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        FLAML offers a novel cost-effective hyperparameter tuning approach.
        It leverages the structure of search space
        to optimize the search order, capable of handling complex constraints/guidance.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} alt={title} />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
