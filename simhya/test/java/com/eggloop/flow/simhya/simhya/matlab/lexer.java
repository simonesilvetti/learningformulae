package com.eggloop.flow.simhya.simhya.matlab;

public class lexer {
    public static void main(String[] args) {
        StringBuilder builder = new StringBuilder();
        builder.append("const double Tl_31=27.450073546391963;\n" +
                "const double Tu_31=67.56277386007622;\n" +
                "const double Tl_29=71.0574965621369;\n" +
                "const double Tu_29=96.67289381767277;\n" +
                "const double Tl_32=30.20357776300566;\n" +
                "const double Tu_32=57.19852321214387;\n" +
                "const double Theta_28=66.10587994676861;\n" +
                "const double Theta_27=0.0;\n" +
                "(F[Tl_31, Tu_31] (flow <= Theta_28) U[Tl_32, Tu_32] !(F[Tl_29, Tu_29] (flow <= Theta_27)))");
    }

}
