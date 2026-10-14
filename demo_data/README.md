# Demo orders

12 hand-picked orders that show every decision rule of the system, used in the video and by the
website's "Run demo orders" button. Rebuild with `python scripts/make_demo_data.py`.

- `orders.csv`: `order_id, category, review_text, image, shows` (empty `image` = order without a photo)
- `images/<order_id>.jpg`: customer photos, named by order id (the same format the Batch page accepts as a ZIP)
- `expected.csv`: the decision the deployed models give for each order

**These orders are simulated.** Each review and photo come from different customers; they were
paired to demonstrate the rules. Reviews come from the text test split and photos from the image test
splits, so no model was trained on them.

## Sources and licences

- Package photos: Roboflow Universe "BoxCondition" (`sih-yna9n/boxcondition-onzuw`), CC BY 4.0.
- Fruit photos: Kaggle "Fruits fresh and rotten for classification" (`sriramr`), licence not stated on Kaggle;
  used here only for a non-commercial student demo.
- Review texts: Amazon Reviews 2023 (McAuley Lab, research dataset) and Flipkart product reviews
  (Kaggle `niraliivaghani`, Open Database licence); short excerpts for illustration.
