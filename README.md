# How to run the experiment

1. The first thing you need to do is downloand the OULA Dataset from [https://analyse.kmi.open.ac.uk/open_dataset](https://analyse.kmi.open.ac.uk/open_dataset) and upload it to the `data` folder. You may need to create the folder. You only need to download the following files:
   - `studentInfo.csv`
   - `studentVle.csv`
   - `vle.csv`
2. Once the OULA Dataset is downloaded you need to run the Get Data cells to prepare all necesary data for the tests. You will be able to store the data. By default it will be stored in the `course_stages_data` folder. You may need to create the folder.
3. After this process is finished, you can run the `statistics` cells to perform an analysis of the data itself. This is an optional step.
4. For running the test you have two scripts:
   - `feature_selection.py` for running the voting feature selection test.
   - `oe_feature_selection.py` for running the genetic feature selection test.
5. The results of the tests will be stored in the corresponding `.csv` file.
6. Finally you can run the `results` cells to review the obtained results in more details.
---

If you have any questions you can contact me at: [alemarti@uji.es](mailto:alemarti@uji.es)

---

This experiment was developed by Alex Martínez-Martínez as part of his research in the Institute of New Imaging Technologies at the Universitat Jaume I.

---

Coauthors: Raul Montoliu and Inmaculada Remolar.

---