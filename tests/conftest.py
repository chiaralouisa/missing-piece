import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def small_study(tmp_path):
    """A miniature cBioPortal study exercising the panel-coverage logic.

    Two panels: SMALL (A, B, C) and BIG (A..F). Samples S1/S2 were sequenced on
    BIG, S3/S4 on SMALL. Genes D/E/F were therefore never interrogated in S3/S4
    -- the loader must mark them unassayed rather than wild-type.
    """
    root = tmp_path / "study"
    root.mkdir()

    (root / "data_gene_panel_SMALL.txt").write_text(
        "stable_id: SMALL\ndescription: small\ngene_list: GENEA\tGENEB\tGENEC\n"
    )
    (root / "data_gene_panel_BIG.txt").write_text(
        "stable_id: BIG\ndescription: big\n"
        "gene_list: GENEA\tGENEB\tGENEC\tGENED\tGENEE\tGENEF\n"
    )

    (root / "data_clinical_sample.txt").write_text(
        "#Sample Id\tPatient Id\tOncotree Code\tSample Type\n"
        "#Sample Id\tPatient Id\tOncotree Code\tSample Type\n"
        "#STRING\tSTRING\tSTRING\tSTRING\n"
        "#1\t1\t1\t1\n"
        "SAMPLE_ID\tPATIENT_ID\tONCOTREE_CODE\tSAMPLE_TYPE\n"
        "S1\tP1\tLUAD\tPrimary\n"
        "S2\tP2\tLUAD\tMetastasis\n"
        "S3\tP3\tLUAD\tPrimary\n"
        "S4\tP4\tLUSC\tPrimary\n"
        "S5\tP1\tLUAD\tMetastasis\n"
    )
    (root / "data_clinical_patient.txt").write_text(
        "#Patient Id\tSex\n#Patient Id\tSex\n#STRING\tSTRING\n#1\t1\n"
        "PATIENT_ID\tSEX\nP1\tFemale\nP2\tMale\nP3\tFemale\nP4\tMale\n"
    )
    (root / "data_gene_panel_matrix.txt").write_text(
        "SAMPLE_ID\tmutations\tcna\n"
        "S1\tBIG\tBIG\nS2\tBIG\tBIG\nS3\tSMALL\tSMALL\nS4\tSMALL\tSMALL\nS5\tBIG\tBIG\n"
    )
    (root / "data_mutations.txt").write_text(
        "Hugo_Symbol\tTumor_Sample_Barcode\tVariant_Classification\n"
        "GENEA\tS1\tMissense_Mutation\n"
        "GENED\tS1\tNonsense_Mutation\n"
        "GENEB\tS2\tSilent\n"
        "GENEE\tS2\tFrame_Shift_Del\n"
        "GENEA\tS3\tMissense_Mutation\n"
        "GENEC\tS4\tSplice_Site\n"
        "GENEA\tS5\tMissense_Mutation\n"
    )
    (root / "data_cna.txt").write_text(
        "Hugo_Symbol\tS1\tS2\tS3\tS4\tS5\n"
        "GENEA\t0\t0\t0\t0\t0\n"
        "GENEB\t0\t-2\t0\t0\t0\n"
        "GENEC\t1\t0\t0\t0\t0\n"
        "GENED\t0\t0\t0\t0\t2\n"
        "GENEE\t0\t0\t0\t0\t0\n"
        "GENEF\t2\t0\t0\t0\t0\n"
    )
    return root
