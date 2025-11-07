import sys
sys.path.insert(0, '.')


if __name__ == '__main__':
    import fire
    from train import main
    fire.Fire(main)
