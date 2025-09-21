# E-Commerce Delivery Status Prediction

## **Project Information**

**Author:** Aravindan Natarajan

**Version:** 1.0

## **License**

MIT License

## **Data Source**

The data set has been downloaded from here: [https://www.kaggle.com/datasets/davidafolayan/e-commerce-dataset](https://www.kaggle.com/datasets/davidafolayan/e-commerce-dataset)

## **Data Description**

This dataset contains comprehensive e-commerce transaction records, including customer details (name, country, region, state, city), geographic coordinates (latitude, longitude), order specifics (order ID, category, product name, shipping type), and delivery metrics (ship date, delivery status, scheduled and real shipment days). It also includes financial metrics such as order discounts, sales, quantity, and profit, providing a holistic view for analysis.

## **Data Dictionary**

| Column Name                 | Description                                                        | Feature/Target   |
| :-------------------------- | :----------------------------------------------------------------- | :--------------- |
| **Order ID**                | A unique identifier for each order.                                | Feature          |
| **Order Date**              | The date the order was placed.                                     | Feature          |
| **Customer Name**           | The name of the customer.                                          | Feature          |
| **Country**                 | The country where the customer resides.                            | Feature          |
| **Region**                  | The region within the country.                                     | Feature          |
| **State**                   | The state within the country/region.                               | Feature          |
| **City**                    | The city within the state.                                         | Feature          |
| **Lat**                     | The latitude coordinate of the customer's location.                | Feature          |
| **Long**                    | The longitude coordinate of the customer's location.               | Feature          |
| **Category**                | The product category of the item ordered.                          | Feature          |
| **Product Name**            | The name of the specific product.                                  | Feature          |
| **Sales**                   | The total sales amount for the order.                              | Feature          |
| **Quantity**                | The number of units of the product ordered.                        | Feature          |
| **Discount**                | The discount applied to the order.                                 | Feature          |
| **Profit**                  | The profit generated from the order.                               | Feature          |
| **Shipping Type**           | The method used for shipping the order.                            | Feature          |
| **Delivery Status**         | The current status of the order delivery.                          | Target           |
| **Scheduled Delivery Date** | The initially scheduled date for delivery.                         | Feature          |
| **Real Delivery Date**      | The actual date the order was delivered.                           | Feature          |

## **Exploratory Data Analysis**

The exploratory data analysis revealed several key insights:

*   **Delivery Status Imbalance:** 'Late delivery' is the most frequent delivery status, indicating a significant operational challenge.
*   **Categorical Features:** 'Office Supplies' is the most popular category, the 'Consumer' segment is the largest customer base, and 'Standard Class' is the most common shipping type. Order volume varies significantly across cities, states, and regions.
*   **Numeric Features:** Financial features (`order_item_discount`, `sales_per_order`, `order_quantity`, `profit_per_order`) show skewed distributions with outliers. `shipment_day_difference` highlights early, on-time, and late shipments.
*   **Temporal Analysis:** Order volume shows a decline in late 2022 and exhibits seasonal patterns throughout the year.

## **Model Building and Evaluation**

Several classification models were trained and evaluated: Logistic Regression, Random Forest, XGBoost, LightGBM, Easy Ensemble, and Balanced Random Forest.

The models were evaluated using stratified k-fold cross-validation, out-of-fold predictions, and an unseen test set, using metrics including Accuracy, Precision (Macro), Recall (Macro), F1-score (Macro), and Matthews Correlation Coefficient (MCC).

| Model                     | CV Accuracy (Mean) | CV Precision (Macro) | CV Recall (Macro) | CV F1-score (Macro) | CV MCC (Mean) | OOF Accuracy | OOF Precision (Macro) | OOF Recall (Macro) | OOF F1-score (Macro) | OOF MCC | Test Accuracy | Test Precision (Macro) | Test Recall (Macro) | Test F1-score (Macro) | Test MCC |
| :------------------------ | :----------------- | :------------------- | :---------------- | :------------------ | :------------ | :----------- | :-------------------- | :----------------- | :------------------- | :------ | :------------ | :--------------------- | :------------------ | :-------------------- | :------- |
| **LightGBM**              | 0.9552             | 0.9557               | 0.7519            | 0.7400              | 0.9274        | 0.9552       | 0.9556                | 0.7519             | 0.7400               | 0.9274  | 0.9549        | 0.9392                 | 0.7536              | 0.7437                | 0.9269   |
| **XGBoost**               | 0.9549             | 0.9226               | 0.7520            | 0.7402              | 0.9268        | 0.9549       | 0.9241                | 0.7520             | 0.7402               | 0.9268  | 0.9548        | 0.9053                 | 0.7536              | 0.7436                | 0.9267   |
| **Random Forest**         | 0.9545             | 0.8525               | 0.7490            | 0.7340              | 0.9262        | 0.9545       | 0.8646                | 0.7490             | 0.7340               | 0.9262  | 0.9542        | 0.8409                 | 0.7491              | 0.7348                | 0.9258   |
| **Balanced Random Forest**| 0.9431             | 0.7479               | 0.7521            | 0.7433              | 0.9070        | 0.9431       | 0.7476                | 0.7521             | 0.7433               | 0.9070  | 0.9433        | 0.7514                 | 0.7540              | 0.7462                | 0.9073   |
| **Logistic Regression**   | 0.8773             | 0.6801               | 0.6462            | 0.6518              | 0.7990        | 0.8773       | 0.6801                | 0.6462             | 0.6519               | 0.7990  | 0.8768        | 0.6814                 | 0.6446              | 0.6509                | 0.7985   |
| **Easy Ensemble**         | 0.4531             | 0.7232               | 0.5100            | 0.4445              | 0.4416        | 0.4531       | 0.7231                | 0.5100             | 0.4495               | 0.4438  | 0.4549        | 0.7231                 | 0.5117              | 0.4495                | 0.4438   |

LightGBM, XGBoost, and Random Forest consistently achieved the highest performance, with MCC scores above 0.92 on the test set. Balanced Random Forest also performed well in handling class imbalance.

## **Explainable AI (XAI)**

**Global Feature Importance (LightGBM):**

*   Financial features (`profit_per_order`, `order_item_discount`, `sales_per_order`) and geographical features (`latitude`, `longitude`) are the most important global factors influencing delivery status.
*   `shipment_day_difference` is also highly important.
*   Temporal features and order quantity have moderate importance.
*   Categorical features like customer segment, category name, and shipping type contribute but have lower global importance.

**Local Feature Importance (LIME):**

*   LIME explanations provide insights into which features contribute to the prediction of a specific delivery status for individual instances, aiding in understanding why a particular order was predicted as late, on time, or canceled.

## **Conclusions**

The analysis highlights the pervasive issue of late deliveries and identifies key contributing factors, including financial aspects of orders, customer location, and the difference between scheduled and real shipment times. The top-performing models can be used to predict delivery status, enabling proactive measures to mitigate delays. The XAI analysis provides valuable insights into the drivers of delivery outcomes, which can inform business strategies for improving logistics, managing customer expectations, and optimizing operations to reduce late deliveries and enhance overall customer satisfaction.
