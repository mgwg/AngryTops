from __future__ import print_function

import h5py

def print_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        print(len(f.attrs.items()))
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items())==0:
            return

        for layer, g in f.items():
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for p_name in g.keys():
                param = g[p_name]
                print("      {}: {}".format(p_name, param.shape))
            print("      Weights:")
            print("               {}".format(layer.params[0]))
    except Exception as e:
        print(e)
    finally:
        f.close()

if __name__ == '__main__':
    print_structure('/Users/fardinsyed/Desktop/Top_Quark_Project/AngryTops/CheckPoints/tests/test3/dense_multi3.3/simple_model.h5')
